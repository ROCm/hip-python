# AMD_COPYRIGHT
from libc.stdint cimport *

cimport hip._util.posixloader as loader


cdef void* _lib_handle = loader.open_library("libamdhip64.so")


cdef void* hipCreateChannelDesc_funptr = NULL
cdef hipChannelFormatDesc hipCreateChannelDesc(int x,int y,int z,int w,hipChannelFormatKind f) nogil:
    global _lib_handle
    global hipCreateChannelDesc_funptr
    if hipCreateChannelDesc_funptr == NULL:
        with gil:
            hipCreateChannelDesc_funptr = loader.load_symbol(_lib_handle, "hipCreateChannelDesc")
    return (<hipChannelFormatDesc (*)(int,int,int,int,hipChannelFormatKind) nogil> hipCreateChannelDesc_funptr)(x,y,z,w,f)


cdef void* hipInit_funptr = NULL
# @defgroup API HIP API
# @{
# Defines the HIP API.  See the individual sections for more information.
# @defgroup Driver Initialization and Version
# @{
# This section describes the initializtion and version functions of HIP runtime API.
# @brief Explicitly initializes the HIP runtime.
# Most HIP APIs implicitly initialize the HIP runtime.
# This API provides control over the timing of the initialization.
cdef hipError_t hipInit(unsigned int flags) nogil:
    global _lib_handle
    global hipInit_funptr
    if hipInit_funptr == NULL:
        with gil:
            hipInit_funptr = loader.load_symbol(_lib_handle, "hipInit")
    return (<hipError_t (*)(unsigned int) nogil> hipInit_funptr)(flags)


cdef void* hipDriverGetVersion_funptr = NULL
# @brief Returns the approximate HIP driver version.
# @param [out] driverVersion
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning The HIP feature set does not correspond to an exact CUDA SDK driver revision.
# This function always set *driverVersion to 4 as an approximation though HIP supports
# some features which were introduced in later CUDA SDK revisions.
# HIP apps code should not rely on the driver revision number here and should
# use arch feature flags to test device capabilities or conditional compilation.
# @see hipRuntimeGetVersion
cdef hipError_t hipDriverGetVersion(int * driverVersion) nogil:
    global _lib_handle
    global hipDriverGetVersion_funptr
    if hipDriverGetVersion_funptr == NULL:
        with gil:
            hipDriverGetVersion_funptr = loader.load_symbol(_lib_handle, "hipDriverGetVersion")
    return (<hipError_t (*)(int *) nogil> hipDriverGetVersion_funptr)(driverVersion)


cdef void* hipRuntimeGetVersion_funptr = NULL
# @brief Returns the approximate HIP Runtime version.
# @param [out] runtimeVersion
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning The version definition of HIP runtime is different from CUDA.
# On AMD platform, the function returns HIP runtime version,
# while on NVIDIA platform, it returns CUDA runtime version.
# And there is no mapping/correlation between HIP version and CUDA version.
# @see hipDriverGetVersion
cdef hipError_t hipRuntimeGetVersion(int * runtimeVersion) nogil:
    global _lib_handle
    global hipRuntimeGetVersion_funptr
    if hipRuntimeGetVersion_funptr == NULL:
        with gil:
            hipRuntimeGetVersion_funptr = loader.load_symbol(_lib_handle, "hipRuntimeGetVersion")
    return (<hipError_t (*)(int *) nogil> hipRuntimeGetVersion_funptr)(runtimeVersion)


cdef void* hipDeviceGet_funptr = NULL
# @brief Returns a handle to a compute device
# @param [out] device
# @param [in] ordinal
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceGet(hipDevice_t * device,int ordinal) nogil:
    global _lib_handle
    global hipDeviceGet_funptr
    if hipDeviceGet_funptr == NULL:
        with gil:
            hipDeviceGet_funptr = loader.load_symbol(_lib_handle, "hipDeviceGet")
    return (<hipError_t (*)(hipDevice_t *,int) nogil> hipDeviceGet_funptr)(device,ordinal)


cdef void* hipDeviceComputeCapability_funptr = NULL
# @brief Returns the compute capability of the device
# @param [out] major
# @param [out] minor
# @param [in] device
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceComputeCapability(int * major,int * minor,hipDevice_t device) nogil:
    global _lib_handle
    global hipDeviceComputeCapability_funptr
    if hipDeviceComputeCapability_funptr == NULL:
        with gil:
            hipDeviceComputeCapability_funptr = loader.load_symbol(_lib_handle, "hipDeviceComputeCapability")
    return (<hipError_t (*)(int *,int *,hipDevice_t) nogil> hipDeviceComputeCapability_funptr)(major,minor,device)


cdef void* hipDeviceGetName_funptr = NULL
# @brief Returns an identifer string for the device.
# @param [out] name
# @param [in] len
# @param [in] device
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceGetName(char * name,int len,hipDevice_t device) nogil:
    global _lib_handle
    global hipDeviceGetName_funptr
    if hipDeviceGetName_funptr == NULL:
        with gil:
            hipDeviceGetName_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetName")
    return (<hipError_t (*)(char *,int,hipDevice_t) nogil> hipDeviceGetName_funptr)(name,len,device)


cdef void* hipDeviceGetUuid_funptr = NULL
# @brief Returns an UUID for the device.[BETA]
# @param [out] uuid
# @param [in] device
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotInitialized,
# #hipErrorDeinitialized
cdef hipError_t hipDeviceGetUuid(hipUUID_t * uuid,hipDevice_t device) nogil:
    global _lib_handle
    global hipDeviceGetUuid_funptr
    if hipDeviceGetUuid_funptr == NULL:
        with gil:
            hipDeviceGetUuid_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetUuid")
    return (<hipError_t (*)(hipUUID_t *,hipDevice_t) nogil> hipDeviceGetUuid_funptr)(uuid,device)


cdef void* hipDeviceGetP2PAttribute_funptr = NULL
# @brief Returns a value for attr of link between two devices
# @param [out] value
# @param [in] attr
# @param [in] srcDevice
# @param [in] dstDevice
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceGetP2PAttribute(int * value,hipDeviceP2PAttr attr,int srcDevice,int dstDevice) nogil:
    global _lib_handle
    global hipDeviceGetP2PAttribute_funptr
    if hipDeviceGetP2PAttribute_funptr == NULL:
        with gil:
            hipDeviceGetP2PAttribute_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetP2PAttribute")
    return (<hipError_t (*)(int *,hipDeviceP2PAttr,int,int) nogil> hipDeviceGetP2PAttribute_funptr)(value,attr,srcDevice,dstDevice)


cdef void* hipDeviceGetPCIBusId_funptr = NULL
# @brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
# @param [out] pciBusId
# @param [in] len
# @param [in] device
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceGetPCIBusId(char * pciBusId,int len,int device) nogil:
    global _lib_handle
    global hipDeviceGetPCIBusId_funptr
    if hipDeviceGetPCIBusId_funptr == NULL:
        with gil:
            hipDeviceGetPCIBusId_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetPCIBusId")
    return (<hipError_t (*)(char *,int,int) nogil> hipDeviceGetPCIBusId_funptr)(pciBusId,len,device)


cdef void* hipDeviceGetByPCIBusId_funptr = NULL
# @brief Returns a handle to a compute device.
# @param [out] device handle
# @param [in] PCI Bus ID
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
cdef hipError_t hipDeviceGetByPCIBusId(int * device,const char * pciBusId) nogil:
    global _lib_handle
    global hipDeviceGetByPCIBusId_funptr
    if hipDeviceGetByPCIBusId_funptr == NULL:
        with gil:
            hipDeviceGetByPCIBusId_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetByPCIBusId")
    return (<hipError_t (*)(int *,const char *) nogil> hipDeviceGetByPCIBusId_funptr)(device,pciBusId)


cdef void* hipDeviceTotalMem_funptr = NULL
# @brief Returns the total amount of memory on the device.
# @param [out] bytes
# @param [in] device
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceTotalMem(int * bytes,hipDevice_t device) nogil:
    global _lib_handle
    global hipDeviceTotalMem_funptr
    if hipDeviceTotalMem_funptr == NULL:
        with gil:
            hipDeviceTotalMem_funptr = loader.load_symbol(_lib_handle, "hipDeviceTotalMem")
    return (<hipError_t (*)(int *,hipDevice_t) nogil> hipDeviceTotalMem_funptr)(bytes,device)


cdef void* hipDeviceSynchronize_funptr = NULL
# @}
# @defgroup Device Device Management
# @{
# This section describes the device management functions of HIP runtime API.
# @brief Waits on all active streams on current device
# When this command is invoked, the host thread gets blocked until all the commands associated
# with streams associated with the device. HIP does not support multiple blocking modes (yet!).
# @returns #hipSuccess
# @see hipSetDevice, hipDeviceReset
cdef hipError_t hipDeviceSynchronize() nogil:
    global _lib_handle
    global hipDeviceSynchronize_funptr
    if hipDeviceSynchronize_funptr == NULL:
        with gil:
            hipDeviceSynchronize_funptr = loader.load_symbol(_lib_handle, "hipDeviceSynchronize")
    return (<hipError_t (*)() nogil> hipDeviceSynchronize_funptr)()


cdef void* hipDeviceReset_funptr = NULL
# @brief The state of current device is discarded and updated to a fresh state.
# Calling this function deletes all streams created, memory allocated, kernels running, events
# created. Make sure that no other thread is using the device or streams, memory, kernels, events
# associated with the current device.
# @returns #hipSuccess
# @see hipDeviceSynchronize
cdef hipError_t hipDeviceReset() nogil:
    global _lib_handle
    global hipDeviceReset_funptr
    if hipDeviceReset_funptr == NULL:
        with gil:
            hipDeviceReset_funptr = loader.load_symbol(_lib_handle, "hipDeviceReset")
    return (<hipError_t (*)() nogil> hipDeviceReset_funptr)()


cdef void* hipSetDevice_funptr = NULL
# @brief Set default device to be used for subsequent hip API calls from this thread.
# @param[in] deviceId Valid device in range 0...hipGetDeviceCount().
# Sets @p device as the default device for the calling host thread.  Valid device id's are 0...
# (hipGetDeviceCount()-1).
# Many HIP APIs implicitly use the "default device" :
# - Any device memory subsequently allocated from this host thread (using hipMalloc) will be
# allocated on device.
# - Any streams or events created from this host thread will be associated with device.
# - Any kernels launched from this host thread (using hipLaunchKernel) will be executed on device
# (unless a specific stream is specified, in which case the device associated with that stream will
# be used).
# This function may be called from any host thread.  Multiple host threads may use the same device.
# This function does no synchronization with the previous or new device, and has very little
# runtime overhead. Applications can use hipSetDevice to quickly switch the default device before
# making a HIP runtime call which uses the default device.
# The default device is stored in thread-local-storage for each thread.
# Thread-pool implementations may inherit the default device of the previous thread.  A good
# practice is to always call hipSetDevice at the start of HIP coding sequency to establish a known
# standard device.
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorDeviceAlreadyInUse
# @see hipGetDevice, hipGetDeviceCount
cdef hipError_t hipSetDevice(int deviceId) nogil:
    global _lib_handle
    global hipSetDevice_funptr
    if hipSetDevice_funptr == NULL:
        with gil:
            hipSetDevice_funptr = loader.load_symbol(_lib_handle, "hipSetDevice")
    return (<hipError_t (*)(int) nogil> hipSetDevice_funptr)(deviceId)


cdef void* hipGetDevice_funptr = NULL
# @brief Return the default device id for the calling host thread.
# @param [out] device *device is written with the default device
# HIP maintains an default device for each thread using thread-local-storage.
# This device is used implicitly for HIP runtime APIs called by this thread.
# hipGetDevice returns in * @p device the default device for the calling host thread.
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @see hipSetDevice, hipGetDevicesizeBytes
cdef hipError_t hipGetDevice(int * deviceId) nogil:
    global _lib_handle
    global hipGetDevice_funptr
    if hipGetDevice_funptr == NULL:
        with gil:
            hipGetDevice_funptr = loader.load_symbol(_lib_handle, "hipGetDevice")
    return (<hipError_t (*)(int *) nogil> hipGetDevice_funptr)(deviceId)


cdef void* hipGetDeviceCount_funptr = NULL
# @brief Return number of compute-capable devices.
# @param [output] count Returns number of compute-capable devices.
# @returns #hipSuccess, #hipErrorNoDevice
# Returns in @p *count the number of devices that have ability to run compute commands.  If there
# are no such devices, then @ref hipGetDeviceCount will return #hipErrorNoDevice. If 1 or more
# devices can be found, then hipGetDeviceCount returns #hipSuccess.
cdef hipError_t hipGetDeviceCount(int * count) nogil:
    global _lib_handle
    global hipGetDeviceCount_funptr
    if hipGetDeviceCount_funptr == NULL:
        with gil:
            hipGetDeviceCount_funptr = loader.load_symbol(_lib_handle, "hipGetDeviceCount")
    return (<hipError_t (*)(int *) nogil> hipGetDeviceCount_funptr)(count)


cdef void* hipDeviceGetAttribute_funptr = NULL
# @brief Query for a specific device attribute.
# @param [out] pi pointer to value to return
# @param [in] attr attribute to query
# @param [in] deviceId which device to query for information
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
cdef hipError_t hipDeviceGetAttribute(int * pi,hipDeviceAttribute_t attr,int deviceId) nogil:
    global _lib_handle
    global hipDeviceGetAttribute_funptr
    if hipDeviceGetAttribute_funptr == NULL:
        with gil:
            hipDeviceGetAttribute_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetAttribute")
    return (<hipError_t (*)(int *,hipDeviceAttribute_t,int) nogil> hipDeviceGetAttribute_funptr)(pi,attr,deviceId)


cdef void* hipDeviceGetDefaultMemPool_funptr = NULL
# @brief Returns the default memory pool of the specified device
# @param [out] mem_pool Default memory pool to return
# @param [in] device    Device index for query the default memory pool
# @returns #chipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotSupported
# @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool,int device) nogil:
    global _lib_handle
    global hipDeviceGetDefaultMemPool_funptr
    if hipDeviceGetDefaultMemPool_funptr == NULL:
        with gil:
            hipDeviceGetDefaultMemPool_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetDefaultMemPool")
    return (<hipError_t (*)(hipMemPool_t*,int) nogil> hipDeviceGetDefaultMemPool_funptr)(mem_pool,device)


cdef void* hipDeviceSetMemPool_funptr = NULL
# @brief Sets the current memory pool of a device
# The memory pool must be local to the specified device.
# @p hipMallocAsync allocates from the current mempool of the provided stream's device.
# By default, a device's current memory pool is its default memory pool.
# @note Use @p hipMallocFromPoolAsync for asynchronous memory allocations from a device
# different than the one the stream runs on.
# @param [in] device   Device index for the update
# @param [in] mem_pool Memory pool for update as the current on the specified device
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice, #hipErrorNotSupported
# @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceSetMemPool(int device,hipMemPool_t mem_pool) nogil:
    global _lib_handle
    global hipDeviceSetMemPool_funptr
    if hipDeviceSetMemPool_funptr == NULL:
        with gil:
            hipDeviceSetMemPool_funptr = loader.load_symbol(_lib_handle, "hipDeviceSetMemPool")
    return (<hipError_t (*)(int,hipMemPool_t) nogil> hipDeviceSetMemPool_funptr)(device,mem_pool)


cdef void* hipDeviceGetMemPool_funptr = NULL
# @brief Gets the current memory pool for the specified device
# Returns the last pool provided to @p hipDeviceSetMemPool for this device
# or the device's default memory pool if @p hipDeviceSetMemPool has never been called.
# By default the current mempool is the default mempool for a device,
# otherwise the returned pool must have been set with @p hipDeviceSetMemPool.
# @param [out] mem_pool Current memory pool on the specified device
# @param [in] device    Device index to query the current memory pool
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool,int device) nogil:
    global _lib_handle
    global hipDeviceGetMemPool_funptr
    if hipDeviceGetMemPool_funptr == NULL:
        with gil:
            hipDeviceGetMemPool_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetMemPool")
    return (<hipError_t (*)(hipMemPool_t*,int) nogil> hipDeviceGetMemPool_funptr)(mem_pool,device)


cdef void* hipGetDeviceProperties_funptr = NULL
# @brief Returns device properties.
# @param [out] prop written with device properties
# @param [in]  deviceId which device to query for information
# @return #hipSuccess, #hipErrorInvalidDevice
# @bug HCC always returns 0 for maxThreadsPerMultiProcessor
# @bug HCC always returns 0 for regsPerBlock
# @bug HCC always returns 0 for l2CacheSize
# Populates hipGetDeviceProperties with information for the specified device.
cdef hipError_t hipGetDeviceProperties(hipDeviceProp_t * prop,int deviceId) nogil:
    global _lib_handle
    global hipGetDeviceProperties_funptr
    if hipGetDeviceProperties_funptr == NULL:
        with gil:
            hipGetDeviceProperties_funptr = loader.load_symbol(_lib_handle, "hipGetDeviceProperties")
    return (<hipError_t (*)(hipDeviceProp_t *,int) nogil> hipGetDeviceProperties_funptr)(prop,deviceId)


cdef void* hipDeviceSetCacheConfig_funptr = NULL
# @brief Set L1/Shared cache partition.
# @param [in] cacheConfig
# @returns #hipSuccess, #hipErrorNotInitialized
# Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
# on those architectures.
cdef hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) nogil:
    global _lib_handle
    global hipDeviceSetCacheConfig_funptr
    if hipDeviceSetCacheConfig_funptr == NULL:
        with gil:
            hipDeviceSetCacheConfig_funptr = loader.load_symbol(_lib_handle, "hipDeviceSetCacheConfig")
    return (<hipError_t (*)(hipFuncCache_t) nogil> hipDeviceSetCacheConfig_funptr)(cacheConfig)


cdef void* hipDeviceGetCacheConfig_funptr = NULL
# @brief Get Cache configuration for a specific Device
# @param [out] cacheConfig
# @returns #hipSuccess, #hipErrorNotInitialized
# Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
# on those architectures.
cdef hipError_t hipDeviceGetCacheConfig(hipFuncCache_t * cacheConfig) nogil:
    global _lib_handle
    global hipDeviceGetCacheConfig_funptr
    if hipDeviceGetCacheConfig_funptr == NULL:
        with gil:
            hipDeviceGetCacheConfig_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetCacheConfig")
    return (<hipError_t (*)(hipFuncCache_t *) nogil> hipDeviceGetCacheConfig_funptr)(cacheConfig)


cdef void* hipDeviceGetLimit_funptr = NULL
# @brief Get Resource limits of current device
# @param [out] pValue
# @param [in]  limit
# @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
# Note: Currently, only hipLimitMallocHeapSize is available
cdef hipError_t hipDeviceGetLimit(int * pValue,hipLimit_t limit) nogil:
    global _lib_handle
    global hipDeviceGetLimit_funptr
    if hipDeviceGetLimit_funptr == NULL:
        with gil:
            hipDeviceGetLimit_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetLimit")
    return (<hipError_t (*)(int *,hipLimit_t) nogil> hipDeviceGetLimit_funptr)(pValue,limit)


cdef void* hipDeviceSetLimit_funptr = NULL
# @brief Set Resource limits of current device
# @param [in] limit
# @param [in] value
# @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
cdef hipError_t hipDeviceSetLimit(hipLimit_t limit,int value) nogil:
    global _lib_handle
    global hipDeviceSetLimit_funptr
    if hipDeviceSetLimit_funptr == NULL:
        with gil:
            hipDeviceSetLimit_funptr = loader.load_symbol(_lib_handle, "hipDeviceSetLimit")
    return (<hipError_t (*)(hipLimit_t,int) nogil> hipDeviceSetLimit_funptr)(limit,value)


cdef void* hipDeviceGetSharedMemConfig_funptr = NULL
# @brief Returns bank width of shared memory for current device
# @param [out] pConfig
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
# Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
cdef hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig * pConfig) nogil:
    global _lib_handle
    global hipDeviceGetSharedMemConfig_funptr
    if hipDeviceGetSharedMemConfig_funptr == NULL:
        with gil:
            hipDeviceGetSharedMemConfig_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetSharedMemConfig")
    return (<hipError_t (*)(hipSharedMemConfig *) nogil> hipDeviceGetSharedMemConfig_funptr)(pConfig)


cdef void* hipGetDeviceFlags_funptr = NULL
# @brief Gets the flags set for current device
# @param [out] flags
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
cdef hipError_t hipGetDeviceFlags(unsigned int * flags) nogil:
    global _lib_handle
    global hipGetDeviceFlags_funptr
    if hipGetDeviceFlags_funptr == NULL:
        with gil:
            hipGetDeviceFlags_funptr = loader.load_symbol(_lib_handle, "hipGetDeviceFlags")
    return (<hipError_t (*)(unsigned int *) nogil> hipGetDeviceFlags_funptr)(flags)


cdef void* hipDeviceSetSharedMemConfig_funptr = NULL
# @brief The bank width of shared memory on current device is set
# @param [in] config
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
# Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
cdef hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) nogil:
    global _lib_handle
    global hipDeviceSetSharedMemConfig_funptr
    if hipDeviceSetSharedMemConfig_funptr == NULL:
        with gil:
            hipDeviceSetSharedMemConfig_funptr = loader.load_symbol(_lib_handle, "hipDeviceSetSharedMemConfig")
    return (<hipError_t (*)(hipSharedMemConfig) nogil> hipDeviceSetSharedMemConfig_funptr)(config)


cdef void* hipSetDeviceFlags_funptr = NULL
# @brief The current device behavior is changed according the flags passed.
# @param [in] flags
# The schedule flags impact how HIP waits for the completion of a command running on a device.
# hipDeviceScheduleSpin         : HIP runtime will actively spin in the thread which submitted the
# work until the command completes.  This offers the lowest latency, but will consume a CPU core
# and may increase power. hipDeviceScheduleYield        : The HIP runtime will yield the CPU to
# system so that other tasks can use it.  This may increase latency to detect the completion but
# will consume less power and is friendlier to other tasks in the system.
# hipDeviceScheduleBlockingSync : On ROCm platform, this is a synonym for hipDeviceScheduleYield.
# hipDeviceScheduleAuto         : Use a hueristic to select between Spin and Yield modes.  If the
# number of HIP contexts is greater than the number of logical processors in the system, use Spin
# scheduling.  Else use Yield scheduling.
# hipDeviceMapHost              : Allow mapping host memory.  On ROCM, this is always allowed and
# the flag is ignored. hipDeviceLmemResizeToMax      : @warning ROCm silently ignores this flag.
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorSetOnActiveProcess
cdef hipError_t hipSetDeviceFlags(unsigned int flags) nogil:
    global _lib_handle
    global hipSetDeviceFlags_funptr
    if hipSetDeviceFlags_funptr == NULL:
        with gil:
            hipSetDeviceFlags_funptr = loader.load_symbol(_lib_handle, "hipSetDeviceFlags")
    return (<hipError_t (*)(unsigned int) nogil> hipSetDeviceFlags_funptr)(flags)


cdef void* hipChooseDevice_funptr = NULL
# @brief Device which matches hipDeviceProp_t is returned
# @param [out] device ID
# @param [in]  device properties pointer
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipChooseDevice(int * device,hipDeviceProp_t * prop) nogil:
    global _lib_handle
    global hipChooseDevice_funptr
    if hipChooseDevice_funptr == NULL:
        with gil:
            hipChooseDevice_funptr = loader.load_symbol(_lib_handle, "hipChooseDevice")
    return (<hipError_t (*)(int *,hipDeviceProp_t *) nogil> hipChooseDevice_funptr)(device,prop)


cdef void* hipExtGetLinkTypeAndHopCount_funptr = NULL
# @brief Returns the link type and hop count between two devices
# @param [in] device1 Ordinal for device1
# @param [in] device2 Ordinal for device2
# @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
# @param [out] hopcount Returns the hop count between the two devices
# Queries and returns the HSA link type and the hop count between the two specified devices.
# @returns #hipSuccess, #hipInvalidDevice, #hipErrorRuntimeOther
cdef hipError_t hipExtGetLinkTypeAndHopCount(int device1,int device2,uint32_t * linktype,uint32_t * hopcount) nogil:
    global _lib_handle
    global hipExtGetLinkTypeAndHopCount_funptr
    if hipExtGetLinkTypeAndHopCount_funptr == NULL:
        with gil:
            hipExtGetLinkTypeAndHopCount_funptr = loader.load_symbol(_lib_handle, "hipExtGetLinkTypeAndHopCount")
    return (<hipError_t (*)(int,int,uint32_t *,uint32_t *) nogil> hipExtGetLinkTypeAndHopCount_funptr)(device1,device2,linktype,hopcount)


cdef void* hipIpcGetMemHandle_funptr = NULL
# @brief Gets an interprocess memory handle for an existing device memory
# allocation
# Takes a pointer to the base of an existing device memory allocation created
# with hipMalloc and exports it for use in another process. This is a
# lightweight operation and may be called multiple times on an allocation
# without adverse effects.
# If a region of memory is freed with hipFree and a subsequent call
# to hipMalloc returns memory with the same device address,
# hipIpcGetMemHandle will return a unique handle for the
# new memory.
# @param handle - Pointer to user allocated hipIpcMemHandle to return
# the handle in.
# @param devPtr - Base pointer to previously allocated device memory
# @returns
# hipSuccess,
# hipErrorInvalidHandle,
# hipErrorOutOfMemory,
# hipErrorMapFailed,
cdef hipError_t hipIpcGetMemHandle(hipIpcMemHandle_st * handle,void * devPtr) nogil:
    global _lib_handle
    global hipIpcGetMemHandle_funptr
    if hipIpcGetMemHandle_funptr == NULL:
        with gil:
            hipIpcGetMemHandle_funptr = loader.load_symbol(_lib_handle, "hipIpcGetMemHandle")
    return (<hipError_t (*)(hipIpcMemHandle_st *,void *) nogil> hipIpcGetMemHandle_funptr)(handle,devPtr)


cdef void* hipIpcOpenMemHandle_funptr = NULL
# @brief Opens an interprocess memory handle exported from another process
# and returns a device pointer usable in the local process.
# Maps memory exported from another process with hipIpcGetMemHandle into
# the current device address space. For contexts on different devices
# hipIpcOpenMemHandle can attempt to enable peer access between the
# devices as if the user called hipDeviceEnablePeerAccess. This behavior is
# controlled by the hipIpcMemLazyEnablePeerAccess flag.
# hipDeviceCanAccessPeer can determine if a mapping is possible.
# Contexts that may open hipIpcMemHandles are restricted in the following way.
# hipIpcMemHandles from each device in a given process may only be opened
# by one context per device per other process.
# Memory returned from hipIpcOpenMemHandle must be freed with
# hipIpcCloseMemHandle.
# Calling hipFree on an exported memory region before calling
# hipIpcCloseMemHandle in the importing context will result in undefined
# behavior.
# @param devPtr - Returned device pointer
# @param handle - hipIpcMemHandle to open
# @param flags  - Flags for this operation. Must be specified as hipIpcMemLazyEnablePeerAccess
# @returns
# hipSuccess,
# hipErrorMapFailed,
# hipErrorInvalidHandle,
# hipErrorTooManyPeers
# @note During multiple processes, using the same memory handle opened by the current context,
# there is no guarantee that the same device poiter will be returned in @p *devPtr.
# This is diffrent from CUDA.
cdef hipError_t hipIpcOpenMemHandle(void ** devPtr,hipIpcMemHandle_st handle,unsigned int flags) nogil:
    global _lib_handle
    global hipIpcOpenMemHandle_funptr
    if hipIpcOpenMemHandle_funptr == NULL:
        with gil:
            hipIpcOpenMemHandle_funptr = loader.load_symbol(_lib_handle, "hipIpcOpenMemHandle")
    return (<hipError_t (*)(void **,hipIpcMemHandle_st,unsigned int) nogil> hipIpcOpenMemHandle_funptr)(devPtr,handle,flags)


cdef void* hipIpcCloseMemHandle_funptr = NULL
# @brief Close memory mapped with hipIpcOpenMemHandle
# Unmaps memory returnd by hipIpcOpenMemHandle. The original allocation
# in the exporting process as well as imported mappings in other processes
# will be unaffected.
# Any resources used to enable peer access will be freed if this is the
# last mapping using them.
# @param devPtr - Device pointer returned by hipIpcOpenMemHandle
# @returns
# hipSuccess,
# hipErrorMapFailed,
# hipErrorInvalidHandle,
cdef hipError_t hipIpcCloseMemHandle(void * devPtr) nogil:
    global _lib_handle
    global hipIpcCloseMemHandle_funptr
    if hipIpcCloseMemHandle_funptr == NULL:
        with gil:
            hipIpcCloseMemHandle_funptr = loader.load_symbol(_lib_handle, "hipIpcCloseMemHandle")
    return (<hipError_t (*)(void *) nogil> hipIpcCloseMemHandle_funptr)(devPtr)


cdef void* hipIpcGetEventHandle_funptr = NULL
# @brief Gets an opaque interprocess handle for an event.
# This opaque handle may be copied into other processes and opened with hipIpcOpenEventHandle.
# Then hipEventRecord, hipEventSynchronize, hipStreamWaitEvent and hipEventQuery may be used in
# either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
# will result in undefined behavior.
# @param[out]  handle Pointer to hipIpcEventHandle to return the opaque event handle
# @param[in]   event  Event allocated with hipEventInterprocess and hipEventDisableTiming flags
# @returns #hipSuccess, #hipErrorInvalidConfiguration, #hipErrorInvalidValue
cdef hipError_t hipIpcGetEventHandle(hipIpcEventHandle_st * handle,hipEvent_t event) nogil:
    global _lib_handle
    global hipIpcGetEventHandle_funptr
    if hipIpcGetEventHandle_funptr == NULL:
        with gil:
            hipIpcGetEventHandle_funptr = loader.load_symbol(_lib_handle, "hipIpcGetEventHandle")
    return (<hipError_t (*)(hipIpcEventHandle_st *,hipEvent_t) nogil> hipIpcGetEventHandle_funptr)(handle,event)


cdef void* hipIpcOpenEventHandle_funptr = NULL
# @brief Opens an interprocess event handles.
# Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle. The returned
# hipEvent_t behaves like a locally created event with the hipEventDisableTiming flag specified. This event
# need be freed with hipEventDestroy. Operations on the imported event after the exported event has been freed
# with hipEventDestroy will result in undefined behavior. If the function is called within the same process where
# handle is returned by hipIpcGetEventHandle, it will return hipErrorInvalidContext.
# @param[out]  event  Pointer to hipEvent_t to return the event
# @param[in]   handle The opaque interprocess handle to open
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext
cdef hipError_t hipIpcOpenEventHandle(hipEvent_t* event,hipIpcEventHandle_st handle) nogil:
    global _lib_handle
    global hipIpcOpenEventHandle_funptr
    if hipIpcOpenEventHandle_funptr == NULL:
        with gil:
            hipIpcOpenEventHandle_funptr = loader.load_symbol(_lib_handle, "hipIpcOpenEventHandle")
    return (<hipError_t (*)(hipEvent_t*,hipIpcEventHandle_st) nogil> hipIpcOpenEventHandle_funptr)(event,handle)


cdef void* hipFuncSetAttribute_funptr = NULL
# @}
# @defgroup Execution Execution Control
# @{
# This section describes the execution control functions of HIP runtime API.
# @brief Set attribute for a specific function
# @param [in] func;
# @param [in] attr;
# @param [in] value;
# @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
# Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
cdef hipError_t hipFuncSetAttribute(const void * func,hipFuncAttribute attr,int value) nogil:
    global _lib_handle
    global hipFuncSetAttribute_funptr
    if hipFuncSetAttribute_funptr == NULL:
        with gil:
            hipFuncSetAttribute_funptr = loader.load_symbol(_lib_handle, "hipFuncSetAttribute")
    return (<hipError_t (*)(const void *,hipFuncAttribute,int) nogil> hipFuncSetAttribute_funptr)(func,attr,value)


cdef void* hipFuncSetCacheConfig_funptr = NULL
# @brief Set Cache configuration for a specific function
# @param [in] config;
# @returns #hipSuccess, #hipErrorNotInitialized
# Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
# on those architectures.
cdef hipError_t hipFuncSetCacheConfig(const void * func,hipFuncCache_t config) nogil:
    global _lib_handle
    global hipFuncSetCacheConfig_funptr
    if hipFuncSetCacheConfig_funptr == NULL:
        with gil:
            hipFuncSetCacheConfig_funptr = loader.load_symbol(_lib_handle, "hipFuncSetCacheConfig")
    return (<hipError_t (*)(const void *,hipFuncCache_t) nogil> hipFuncSetCacheConfig_funptr)(func,config)


cdef void* hipFuncSetSharedMemConfig_funptr = NULL
# @brief Set shared memory configuation for a specific function
# @param [in] func
# @param [in] config
# @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
# Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
cdef hipError_t hipFuncSetSharedMemConfig(const void * func,hipSharedMemConfig config) nogil:
    global _lib_handle
    global hipFuncSetSharedMemConfig_funptr
    if hipFuncSetSharedMemConfig_funptr == NULL:
        with gil:
            hipFuncSetSharedMemConfig_funptr = loader.load_symbol(_lib_handle, "hipFuncSetSharedMemConfig")
    return (<hipError_t (*)(const void *,hipSharedMemConfig) nogil> hipFuncSetSharedMemConfig_funptr)(func,config)


cdef void* hipGetLastError_funptr = NULL
# @}
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# @defgroup Error Error Handling
# @{
# This section describes the error handling functions of HIP runtime API.
# @brief Return last error returned by any HIP runtime API call and resets the stored error code to
# #hipSuccess
# @returns return code from last HIP called from the active host thread
# Returns the last error that has been returned by any of the runtime calls in the same host
# thread, and then resets the saved error to #hipSuccess.
# @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
cdef hipError_t hipGetLastError() nogil:
    global _lib_handle
    global hipGetLastError_funptr
    if hipGetLastError_funptr == NULL:
        with gil:
            hipGetLastError_funptr = loader.load_symbol(_lib_handle, "hipGetLastError")
    return (<hipError_t (*)() nogil> hipGetLastError_funptr)()


cdef void* hipPeekAtLastError_funptr = NULL
# @brief Return last error returned by any HIP runtime API call.
# @return #hipSuccess
# Returns the last error that has been returned by any of the runtime calls in the same host
# thread. Unlike hipGetLastError, this function does not reset the saved error code.
# @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
cdef hipError_t hipPeekAtLastError() nogil:
    global _lib_handle
    global hipPeekAtLastError_funptr
    if hipPeekAtLastError_funptr == NULL:
        with gil:
            hipPeekAtLastError_funptr = loader.load_symbol(_lib_handle, "hipPeekAtLastError")
    return (<hipError_t (*)() nogil> hipPeekAtLastError_funptr)()


cdef void* hipGetErrorName_funptr = NULL
# @brief Return hip error as text string form.
# @param hip_error Error code to convert to name.
# @return const char pointer to the NULL-terminated error name
# @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
cdef const char * hipGetErrorName(hipError_t hip_error) nogil:
    global _lib_handle
    global hipGetErrorName_funptr
    if hipGetErrorName_funptr == NULL:
        with gil:
            hipGetErrorName_funptr = loader.load_symbol(_lib_handle, "hipGetErrorName")
    return (<const char * (*)(hipError_t) nogil> hipGetErrorName_funptr)(hip_error)


cdef void* hipGetErrorString_funptr = NULL
# @brief Return handy text string message to explain the error which occurred
# @param hipError Error code to convert to string.
# @return const char pointer to the NULL-terminated error string
# @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
cdef const char * hipGetErrorString(hipError_t hipError) nogil:
    global _lib_handle
    global hipGetErrorString_funptr
    if hipGetErrorString_funptr == NULL:
        with gil:
            hipGetErrorString_funptr = loader.load_symbol(_lib_handle, "hipGetErrorString")
    return (<const char * (*)(hipError_t) nogil> hipGetErrorString_funptr)(hipError)


cdef void* hipDrvGetErrorName_funptr = NULL
# @brief Return hip error as text string form.
# @param [in] hipError Error code to convert to string.
# @param [out] const char pointer to the NULL-terminated error string
# @return #hipSuccess, #hipErrorInvalidValue
# @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
cdef hipError_t hipDrvGetErrorName(hipError_t hipError,const char ** errorString) nogil:
    global _lib_handle
    global hipDrvGetErrorName_funptr
    if hipDrvGetErrorName_funptr == NULL:
        with gil:
            hipDrvGetErrorName_funptr = loader.load_symbol(_lib_handle, "hipDrvGetErrorName")
    return (<hipError_t (*)(hipError_t,const char **) nogil> hipDrvGetErrorName_funptr)(hipError,errorString)


cdef void* hipDrvGetErrorString_funptr = NULL
# @brief Return handy text string message to explain the error which occurred
# @param [in] hipError Error code to convert to string.
# @param [out] const char pointer to the NULL-terminated error string
# @return #hipSuccess, #hipErrorInvalidValue
# @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
cdef hipError_t hipDrvGetErrorString(hipError_t hipError,const char ** errorString) nogil:
    global _lib_handle
    global hipDrvGetErrorString_funptr
    if hipDrvGetErrorString_funptr == NULL:
        with gil:
            hipDrvGetErrorString_funptr = loader.load_symbol(_lib_handle, "hipDrvGetErrorString")
    return (<hipError_t (*)(hipError_t,const char **) nogil> hipDrvGetErrorString_funptr)(hipError,errorString)


cdef void* hipStreamCreate_funptr = NULL
# @brief Create an asynchronous stream.
# @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
# newly created stream.
# @return #hipSuccess, #hipErrorInvalidValue
# Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
# reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
# the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
# used by the stream, applicaiton must call hipStreamDestroy.
# @return #hipSuccess, #hipErrorInvalidValue
# @see hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipStreamCreate(hipStream_t* stream) nogil:
    global _lib_handle
    global hipStreamCreate_funptr
    if hipStreamCreate_funptr == NULL:
        with gil:
            hipStreamCreate_funptr = loader.load_symbol(_lib_handle, "hipStreamCreate")
    return (<hipError_t (*)(hipStream_t*) nogil> hipStreamCreate_funptr)(stream)


cdef void* hipStreamCreateWithFlags_funptr = NULL
# @brief Create an asynchronous stream.
# @param[in, out] stream Pointer to new stream
# @param[in ] flags to control stream creation.
# @return #hipSuccess, #hipErrorInvalidValue
# Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
# reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
# the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
# used by the stream, applicaiton must call hipStreamDestroy. Flags controls behavior of the
# stream.  See #hipStreamDefault, #hipStreamNonBlocking.
# @see hipStreamCreate, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipStreamCreateWithFlags(hipStream_t* stream,unsigned int flags) nogil:
    global _lib_handle
    global hipStreamCreateWithFlags_funptr
    if hipStreamCreateWithFlags_funptr == NULL:
        with gil:
            hipStreamCreateWithFlags_funptr = loader.load_symbol(_lib_handle, "hipStreamCreateWithFlags")
    return (<hipError_t (*)(hipStream_t*,unsigned int) nogil> hipStreamCreateWithFlags_funptr)(stream,flags)


cdef void* hipStreamCreateWithPriority_funptr = NULL
# @brief Create an asynchronous stream with the specified priority.
# @param[in, out] stream Pointer to new stream
# @param[in ] flags to control stream creation.
# @param[in ] priority of the stream. Lower numbers represent higher priorities.
# @return #hipSuccess, #hipErrorInvalidValue
# Create a new asynchronous stream with the specified priority.  @p stream returns an opaque handle
# that can be used to reference the newly created stream in subsequent hipStream* commands.  The
# stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
# To release the memory used by the stream, applicaiton must call hipStreamDestroy. Flags controls
# behavior of the stream.  See #hipStreamDefault, #hipStreamNonBlocking.
# @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipStreamCreateWithPriority(hipStream_t* stream,unsigned int flags,int priority) nogil:
    global _lib_handle
    global hipStreamCreateWithPriority_funptr
    if hipStreamCreateWithPriority_funptr == NULL:
        with gil:
            hipStreamCreateWithPriority_funptr = loader.load_symbol(_lib_handle, "hipStreamCreateWithPriority")
    return (<hipError_t (*)(hipStream_t*,unsigned int,int) nogil> hipStreamCreateWithPriority_funptr)(stream,flags,priority)


cdef void* hipDeviceGetStreamPriorityRange_funptr = NULL
# @brief Returns numerical values that correspond to the least and greatest stream priority.
# @param[in, out] leastPriority pointer in which value corresponding to least priority is returned.
# @param[in, out] greatestPriority pointer in which value corresponding to greatest priority is returned.
# Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least
# and greatest stream priority respectively. Stream priorities follow a convention where lower numbers
# imply greater priorities. The range of meaningful stream priorities is given by
# [*greatestPriority, *leastPriority]. If the user attempts to create a stream with a priority value
# that is outside the the meaningful range as specified by this API, the priority is automatically
# clamped to within the valid range.
cdef hipError_t hipDeviceGetStreamPriorityRange(int * leastPriority,int * greatestPriority) nogil:
    global _lib_handle
    global hipDeviceGetStreamPriorityRange_funptr
    if hipDeviceGetStreamPriorityRange_funptr == NULL:
        with gil:
            hipDeviceGetStreamPriorityRange_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetStreamPriorityRange")
    return (<hipError_t (*)(int *,int *) nogil> hipDeviceGetStreamPriorityRange_funptr)(leastPriority,greatestPriority)


cdef void* hipStreamDestroy_funptr = NULL
# @brief Destroys the specified stream.
# @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
# newly created stream.
# @return #hipSuccess #hipErrorInvalidHandle
# Destroys the specified stream.
# If commands are still executing on the specified stream, some may complete execution before the
# queue is deleted.
# The queue may be destroyed while some commands are still inflight, or may wait for all commands
# queued to the stream before destroying it.
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamQuery, hipStreamWaitEvent,
# hipStreamSynchronize
cdef hipError_t hipStreamDestroy(hipStream_t stream) nogil:
    global _lib_handle
    global hipStreamDestroy_funptr
    if hipStreamDestroy_funptr == NULL:
        with gil:
            hipStreamDestroy_funptr = loader.load_symbol(_lib_handle, "hipStreamDestroy")
    return (<hipError_t (*)(hipStream_t) nogil> hipStreamDestroy_funptr)(stream)


cdef void* hipStreamQuery_funptr = NULL
# @brief Return #hipSuccess if all of the operations in the specified @p stream have completed, or
# #hipErrorNotReady if not.
# @param[in] stream stream to query
# @return #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle
# This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
# host threads are sending work to the stream, the status may change immediately after the function
# is called.  It is typically used for debug.
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamSynchronize,
# hipStreamDestroy
cdef hipError_t hipStreamQuery(hipStream_t stream) nogil:
    global _lib_handle
    global hipStreamQuery_funptr
    if hipStreamQuery_funptr == NULL:
        with gil:
            hipStreamQuery_funptr = loader.load_symbol(_lib_handle, "hipStreamQuery")
    return (<hipError_t (*)(hipStream_t) nogil> hipStreamQuery_funptr)(stream)


cdef void* hipStreamSynchronize_funptr = NULL
# @brief Wait for all commands in stream to complete.
# @param[in] stream stream identifier.
# @return #hipSuccess, #hipErrorInvalidHandle
# This command is host-synchronous : the host will block until the specified stream is empty.
# This command follows standard null-stream semantics.  Specifically, specifying the null stream
# will cause the command to wait for other streams on the same device to complete all pending
# operations.
# This command honors the hipDeviceLaunchBlocking flag, which controls whether the wait is active
# or blocking.
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipStreamSynchronize(hipStream_t stream) nogil:
    global _lib_handle
    global hipStreamSynchronize_funptr
    if hipStreamSynchronize_funptr == NULL:
        with gil:
            hipStreamSynchronize_funptr = loader.load_symbol(_lib_handle, "hipStreamSynchronize")
    return (<hipError_t (*)(hipStream_t) nogil> hipStreamSynchronize_funptr)(stream)


cdef void* hipStreamWaitEvent_funptr = NULL
# @brief Make the specified compute stream wait for an event
# @param[in] stream stream to make wait.
# @param[in] event event to wait on
# @param[in] flags control operation [must be 0]
# @return #hipSuccess, #hipErrorInvalidHandle
# This function inserts a wait operation into the specified stream.
# All future work submitted to @p stream will wait until @p event reports completion before
# beginning execution.
# This function only waits for commands in the current stream to complete.  Notably,, this function
# does not impliciy wait for commands in the default stream to complete, even if the specified
# stream is created with hipStreamNonBlocking = 0.
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamDestroy
cdef hipError_t hipStreamWaitEvent(hipStream_t stream,hipEvent_t event,unsigned int flags) nogil:
    global _lib_handle
    global hipStreamWaitEvent_funptr
    if hipStreamWaitEvent_funptr == NULL:
        with gil:
            hipStreamWaitEvent_funptr = loader.load_symbol(_lib_handle, "hipStreamWaitEvent")
    return (<hipError_t (*)(hipStream_t,hipEvent_t,unsigned int) nogil> hipStreamWaitEvent_funptr)(stream,event,flags)


cdef void* hipStreamGetFlags_funptr = NULL
# @brief Return flags associated with this stream.
# @param[in] stream stream to be queried
# @param[in,out] flags Pointer to an unsigned integer in which the stream's flags are returned
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
# @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
# Return flags associated with this stream in *@p flags.
# @see hipStreamCreateWithFlags
cdef hipError_t hipStreamGetFlags(hipStream_t stream,unsigned int * flags) nogil:
    global _lib_handle
    global hipStreamGetFlags_funptr
    if hipStreamGetFlags_funptr == NULL:
        with gil:
            hipStreamGetFlags_funptr = loader.load_symbol(_lib_handle, "hipStreamGetFlags")
    return (<hipError_t (*)(hipStream_t,unsigned int *) nogil> hipStreamGetFlags_funptr)(stream,flags)


cdef void* hipStreamGetPriority_funptr = NULL
# @brief Query the priority of a stream.
# @param[in] stream stream to be queried
# @param[in,out] priority Pointer to an unsigned integer in which the stream's priority is returned
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
# @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
# Query the priority of a stream. The priority is returned in in priority.
# @see hipStreamCreateWithFlags
cdef hipError_t hipStreamGetPriority(hipStream_t stream,int * priority) nogil:
    global _lib_handle
    global hipStreamGetPriority_funptr
    if hipStreamGetPriority_funptr == NULL:
        with gil:
            hipStreamGetPriority_funptr = loader.load_symbol(_lib_handle, "hipStreamGetPriority")
    return (<hipError_t (*)(hipStream_t,int *) nogil> hipStreamGetPriority_funptr)(stream,priority)


cdef void* hipExtStreamCreateWithCUMask_funptr = NULL
# @brief Create an asynchronous stream with the specified CU mask.
# @param[in, out] stream Pointer to new stream
# @param[in ] cuMaskSize Size of CU mask bit array passed in.
# @param[in ] cuMask Bit-vector representing the CU mask. Each active bit represents using one CU.
# The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
# CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are ignored.
# It is user's responsibility to make sure the input is meaningful.
# @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
# Create a new asynchronous stream with the specified CU mask.  @p stream returns an opaque handle
# that can be used to reference the newly created stream in subsequent hipStream* commands.  The
# stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
# To release the memory used by the stream, application must call hipStreamDestroy.
# @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream,uint32_t cuMaskSize,uint32_t * cuMask) nogil:
    global _lib_handle
    global hipExtStreamCreateWithCUMask_funptr
    if hipExtStreamCreateWithCUMask_funptr == NULL:
        with gil:
            hipExtStreamCreateWithCUMask_funptr = loader.load_symbol(_lib_handle, "hipExtStreamCreateWithCUMask")
    return (<hipError_t (*)(hipStream_t*,uint32_t,uint32_t *) nogil> hipExtStreamCreateWithCUMask_funptr)(stream,cuMaskSize,cuMask)


cdef void* hipExtStreamGetCUMask_funptr = NULL
# @brief Get CU mask associated with an asynchronous stream
# @param[in] stream stream to be queried
# @param[in] cuMaskSize number of the block of memories (uint32_t *) allocated by user
# @param[out] cuMask Pointer to a pre-allocated block of memories (uint32_t *) in which
# the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
# each active bit represents one active CU
# @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
# @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipExtStreamGetCUMask(hipStream_t stream,uint32_t cuMaskSize,uint32_t * cuMask) nogil:
    global _lib_handle
    global hipExtStreamGetCUMask_funptr
    if hipExtStreamGetCUMask_funptr == NULL:
        with gil:
            hipExtStreamGetCUMask_funptr = loader.load_symbol(_lib_handle, "hipExtStreamGetCUMask")
    return (<hipError_t (*)(hipStream_t,uint32_t,uint32_t *) nogil> hipExtStreamGetCUMask_funptr)(stream,cuMaskSize,cuMask)


cdef void* hipStreamAddCallback_funptr = NULL
# @brief Adds a callback to be called on the host after all currently enqueued
# items in the stream have completed.  For each
# hipStreamAddCallback call, a callback will be executed exactly once.
# The callback will block later work in the stream until it is finished.
# @param[in] stream   - Stream to add callback to
# @param[in] callback - The function to call once preceding stream operations are complete
# @param[in] userData - User specified data to be passed to the callback function
# @param[in] flags    - Reserved for future use, must be 0
# @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorNotSupported
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamSynchronize,
# hipStreamWaitEvent, hipStreamDestroy, hipStreamCreateWithPriority
cdef hipError_t hipStreamAddCallback(hipStream_t stream,hipStreamCallback_t callback,void * userData,unsigned int flags) nogil:
    global _lib_handle
    global hipStreamAddCallback_funptr
    if hipStreamAddCallback_funptr == NULL:
        with gil:
            hipStreamAddCallback_funptr = loader.load_symbol(_lib_handle, "hipStreamAddCallback")
    return (<hipError_t (*)(hipStream_t,hipStreamCallback_t,void *,unsigned int) nogil> hipStreamAddCallback_funptr)(stream,callback,userData,flags)


cdef void* hipStreamWaitValue32_funptr = NULL
# @}
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# @defgroup StreamM Stream Memory Operations
# @{
# This section describes Stream Memory Wait and Write functions of HIP runtime API.
# @brief Enqueues a wait command to the stream.[BETA]
# @param [in] stream - Stream identifier
# @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
# @param [in] value  - Value to be used in compare operation
# @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
# hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor
# @param [in] mask   - Mask to be applied on value at memory before it is compared with value,
# default value is set to enable every bit
# @returns #hipSuccess, #hipErrorInvalidValue
# Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
# not execute until the defined wait condition is true.
# hipStreamWaitValueGte: waits until *ptr&mask >= value
# hipStreamWaitValueEq : waits until *ptr&mask == value
# hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
# hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
# @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
# @note Support for hipStreamWaitValue32 can be queried using 'hipDeviceGetAttribute()' and
# 'hipDeviceAttributeCanUseStreamWaitValue' flag.
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue64, hipStreamWriteValue64,
# hipStreamWriteValue32, hipDeviceGetAttribute
cdef hipError_t hipStreamWaitValue32(hipStream_t stream,void * ptr,uint32_t value,unsigned int flags,uint32_t mask) nogil:
    global _lib_handle
    global hipStreamWaitValue32_funptr
    if hipStreamWaitValue32_funptr == NULL:
        with gil:
            hipStreamWaitValue32_funptr = loader.load_symbol(_lib_handle, "hipStreamWaitValue32")
    return (<hipError_t (*)(hipStream_t,void *,uint32_t,unsigned int,uint32_t) nogil> hipStreamWaitValue32_funptr)(stream,ptr,value,flags,mask)


cdef void* hipStreamWaitValue64_funptr = NULL
# @brief Enqueues a wait command to the stream.[BETA]
# @param [in] stream - Stream identifier
# @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
# @param [in] value  - Value to be used in compare operation
# @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
# hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor.
# @param [in] mask   - Mask to be applied on value at memory before it is compared with value
# default value is set to enable every bit
# @returns #hipSuccess, #hipErrorInvalidValue
# Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
# not execute until the defined wait condition is true.
# hipStreamWaitValueGte: waits until *ptr&mask >= value
# hipStreamWaitValueEq : waits until *ptr&mask == value
# hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
# hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
# @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
# @note Support for hipStreamWaitValue64 can be queried using 'hipDeviceGetAttribute()' and
# 'hipDeviceAttributeCanUseStreamWaitValue' flag.
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue32, hipStreamWriteValue64,
# hipStreamWriteValue32, hipDeviceGetAttribute
cdef hipError_t hipStreamWaitValue64(hipStream_t stream,void * ptr,uint64_t value,unsigned int flags,uint64_t mask) nogil:
    global _lib_handle
    global hipStreamWaitValue64_funptr
    if hipStreamWaitValue64_funptr == NULL:
        with gil:
            hipStreamWaitValue64_funptr = loader.load_symbol(_lib_handle, "hipStreamWaitValue64")
    return (<hipError_t (*)(hipStream_t,void *,uint64_t,unsigned int,uint64_t) nogil> hipStreamWaitValue64_funptr)(stream,ptr,value,flags,mask)


cdef void* hipStreamWriteValue32_funptr = NULL
# @brief Enqueues a write command to the stream.[BETA]
# @param [in] stream - Stream identifier
# @param [in] ptr    - Pointer to a GPU accessible memory object
# @param [in] value  - Value to be written
# @param [in] flags  - reserved, ignored for now, will be used in future releases
# @returns #hipSuccess, #hipErrorInvalidValue
# Enqueues a write command to the stream, write operation is performed after all earlier commands
# on this stream have completed the execution.
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
# hipStreamWaitValue64
cdef hipError_t hipStreamWriteValue32(hipStream_t stream,void * ptr,uint32_t value,unsigned int flags) nogil:
    global _lib_handle
    global hipStreamWriteValue32_funptr
    if hipStreamWriteValue32_funptr == NULL:
        with gil:
            hipStreamWriteValue32_funptr = loader.load_symbol(_lib_handle, "hipStreamWriteValue32")
    return (<hipError_t (*)(hipStream_t,void *,uint32_t,unsigned int) nogil> hipStreamWriteValue32_funptr)(stream,ptr,value,flags)


cdef void* hipStreamWriteValue64_funptr = NULL
# @brief Enqueues a write command to the stream.[BETA]
# @param [in] stream - Stream identifier
# @param [in] ptr    - Pointer to a GPU accessible memory object
# @param [in] value  - Value to be written
# @param [in] flags  - reserved, ignored for now, will be used in future releases
# @returns #hipSuccess, #hipErrorInvalidValue
# Enqueues a write command to the stream, write operation is performed after all earlier commands
# on this stream have completed the execution.
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
# hipStreamWaitValue64
cdef hipError_t hipStreamWriteValue64(hipStream_t stream,void * ptr,uint64_t value,unsigned int flags) nogil:
    global _lib_handle
    global hipStreamWriteValue64_funptr
    if hipStreamWriteValue64_funptr == NULL:
        with gil:
            hipStreamWriteValue64_funptr = loader.load_symbol(_lib_handle, "hipStreamWriteValue64")
    return (<hipError_t (*)(hipStream_t,void *,uint64_t,unsigned int) nogil> hipStreamWriteValue64_funptr)(stream,ptr,value,flags)


cdef void* hipEventCreateWithFlags_funptr = NULL
# @}
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# @defgroup Event Event Management
# @{
# This section describes the event management functions of HIP runtime API.
# @brief Create an event with the specified flags
# @param[in,out] event Returns the newly created event.
# @param[in] flags     Flags to control event behavior.  Valid values are #hipEventDefault,
#  #hipEventBlockingSync, #hipEventDisableTiming, #hipEventInterprocess
# #hipEventDefault : Default flag.  The event will use active synchronization and will support
#  timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a
#  CPU to poll on the event.
# #hipEventBlockingSync : The event will use blocking synchronization : if hipEventSynchronize is
#  called on this event, the thread will block until the event completes.  This can increase latency
#  for the synchroniation but can result in lower power and more resources for other CPU threads.
# #hipEventDisableTiming : Disable recording of timing information. Events created with this flag
#  would not record profiling data and provide best performance if used for synchronization.
# #hipEventInterprocess : The event can be used as an interprocess event. hipEventDisableTiming
#  flag also must be set when hipEventInterprocess flag is set.
# @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
#  #hipErrorLaunchFailure, #hipErrorOutOfMemory
# @see hipEventCreate, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
cdef hipError_t hipEventCreateWithFlags(hipEvent_t* event,unsigned int flags) nogil:
    global _lib_handle
    global hipEventCreateWithFlags_funptr
    if hipEventCreateWithFlags_funptr == NULL:
        with gil:
            hipEventCreateWithFlags_funptr = loader.load_symbol(_lib_handle, "hipEventCreateWithFlags")
    return (<hipError_t (*)(hipEvent_t*,unsigned int) nogil> hipEventCreateWithFlags_funptr)(event,flags)


cdef void* hipEventCreate_funptr = NULL
# Create an event
# @param[in,out] event Returns the newly created event.
# @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
# #hipErrorLaunchFailure, #hipErrorOutOfMemory
# @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize,
# hipEventDestroy, hipEventElapsedTime
cdef hipError_t hipEventCreate(hipEvent_t* event) nogil:
    global _lib_handle
    global hipEventCreate_funptr
    if hipEventCreate_funptr == NULL:
        with gil:
            hipEventCreate_funptr = loader.load_symbol(_lib_handle, "hipEventCreate")
    return (<hipError_t (*)(hipEvent_t*) nogil> hipEventCreate_funptr)(event)


cdef void* hipEventRecord_funptr = NULL
cdef hipError_t hipEventRecord(hipEvent_t event,hipStream_t stream) nogil:
    global _lib_handle
    global hipEventRecord_funptr
    if hipEventRecord_funptr == NULL:
        with gil:
            hipEventRecord_funptr = loader.load_symbol(_lib_handle, "hipEventRecord")
    return (<hipError_t (*)(hipEvent_t,hipStream_t) nogil> hipEventRecord_funptr)(event,stream)


cdef void* hipEventDestroy_funptr = NULL
# @brief Destroy the specified event.
# @param[in] event Event to destroy.
# @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
# #hipErrorLaunchFailure
# Releases memory associated with the event.  If the event is recording but has not completed
# recording when hipEventDestroy() is called, the function will return immediately and the
# completion_future resources will be released later, when the hipDevice is synchronized.
# @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize, hipEventRecord,
# hipEventElapsedTime
# @returns #hipSuccess
cdef hipError_t hipEventDestroy(hipEvent_t event) nogil:
    global _lib_handle
    global hipEventDestroy_funptr
    if hipEventDestroy_funptr == NULL:
        with gil:
            hipEventDestroy_funptr = loader.load_symbol(_lib_handle, "hipEventDestroy")
    return (<hipError_t (*)(hipEvent_t) nogil> hipEventDestroy_funptr)(event)


cdef void* hipEventSynchronize_funptr = NULL
# @brief Wait for an event to complete.
# This function will block until the event is ready, waiting for all previous work in the stream
# specified when event was recorded with hipEventRecord().
# If hipEventRecord() has not been called on @p event, this function returns immediately.
# TODO-hip- This function needs to support hipEventBlockingSync parameter.
# @param[in] event Event on which to wait.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized,
# #hipErrorInvalidHandle, #hipErrorLaunchFailure
# @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
# hipEventElapsedTime
cdef hipError_t hipEventSynchronize(hipEvent_t event) nogil:
    global _lib_handle
    global hipEventSynchronize_funptr
    if hipEventSynchronize_funptr == NULL:
        with gil:
            hipEventSynchronize_funptr = loader.load_symbol(_lib_handle, "hipEventSynchronize")
    return (<hipError_t (*)(hipEvent_t) nogil> hipEventSynchronize_funptr)(event)


cdef void* hipEventElapsedTime_funptr = NULL
# @brief Return the elapsed time between two events.
# @param[out] ms : Return time between start and stop in ms.
# @param[in]   start : Start event.
# @param[in]   stop  : Stop event.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotReady, #hipErrorInvalidHandle,
# #hipErrorNotInitialized, #hipErrorLaunchFailure
# Computes the elapsed time between two events. Time is computed in ms, with
# a resolution of approximately 1 us.
# Events which are recorded in a NULL stream will block until all commands
# on all other streams complete execution, and then record the timestamp.
# Events which are recorded in a non-NULL stream will record their timestamp
# when they reach the head of the specified stream, after all previous
# commands in that stream have completed executing.  Thus the time that
# the event recorded may be significantly after the host calls hipEventRecord().
# If hipEventRecord() has not been called on either event, then #hipErrorInvalidHandle is
# returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
# recorded on one or both events (that is, hipEventQuery() would return #hipErrorNotReady on at
# least one of the events), then #hipErrorNotReady is returned.
# @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
# hipEventSynchronize
cdef hipError_t hipEventElapsedTime(float * ms,hipEvent_t start,hipEvent_t stop) nogil:
    global _lib_handle
    global hipEventElapsedTime_funptr
    if hipEventElapsedTime_funptr == NULL:
        with gil:
            hipEventElapsedTime_funptr = loader.load_symbol(_lib_handle, "hipEventElapsedTime")
    return (<hipError_t (*)(float *,hipEvent_t,hipEvent_t) nogil> hipEventElapsedTime_funptr)(ms,start,stop)


cdef void* hipEventQuery_funptr = NULL
# @brief Query event status
# @param[in] event Event to query.
# @returns #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle, #hipErrorInvalidValue,
# #hipErrorNotInitialized, #hipErrorLaunchFailure
# Query the status of the specified event.  This function will return #hipSuccess if all
# commands in the appropriate stream (specified to hipEventRecord()) have completed.  If that work
# has not completed, or if hipEventRecord() was not called on the event, then #hipErrorNotReady is
# returned.
# @see hipEventCreate, hipEventCreateWithFlags, hipEventRecord, hipEventDestroy,
# hipEventSynchronize, hipEventElapsedTime
cdef hipError_t hipEventQuery(hipEvent_t event) nogil:
    global _lib_handle
    global hipEventQuery_funptr
    if hipEventQuery_funptr == NULL:
        with gil:
            hipEventQuery_funptr = loader.load_symbol(_lib_handle, "hipEventQuery")
    return (<hipError_t (*)(hipEvent_t) nogil> hipEventQuery_funptr)(event)


cdef void* hipPointerGetAttributes_funptr = NULL
# @}
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# @defgroup Memory Memory Management
# @{
# This section describes the memory management functions of HIP runtime API.
# The following CUDA APIs are not currently supported:
# - cudaMalloc3D
# - cudaMalloc3DArray
# - TODO - more 2D, 3D, array APIs here.
# @brief Return attributes for the specified pointer
# @param [out]  attributes  attributes for the specified pointer
# @param [in]   ptr         pointer to get attributes for
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @see hipPointerGetAttribute
cdef hipError_t hipPointerGetAttributes(hipPointerAttribute_t * attributes,const void * ptr) nogil:
    global _lib_handle
    global hipPointerGetAttributes_funptr
    if hipPointerGetAttributes_funptr == NULL:
        with gil:
            hipPointerGetAttributes_funptr = loader.load_symbol(_lib_handle, "hipPointerGetAttributes")
    return (<hipError_t (*)(hipPointerAttribute_t *,const void *) nogil> hipPointerGetAttributes_funptr)(attributes,ptr)


cdef void* hipPointerGetAttribute_funptr = NULL
# @brief Returns information about the specified pointer.[BETA]
# @param [in, out] data     returned pointer attribute value
# @param [in]      atribute attribute to query for
# @param [in]      ptr      pointer to get attributes for
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# @see hipPointerGetAttributes
cdef hipError_t hipPointerGetAttribute(void * data,hipPointer_attribute attribute,hipDeviceptr_t ptr) nogil:
    global _lib_handle
    global hipPointerGetAttribute_funptr
    if hipPointerGetAttribute_funptr == NULL:
        with gil:
            hipPointerGetAttribute_funptr = loader.load_symbol(_lib_handle, "hipPointerGetAttribute")
    return (<hipError_t (*)(void *,hipPointer_attribute,hipDeviceptr_t) nogil> hipPointerGetAttribute_funptr)(data,attribute,ptr)


cdef void* hipDrvPointerGetAttributes_funptr = NULL
# @brief Returns information about the specified pointer.[BETA]
# @param [in]  numAttributes   number of attributes to query for
# @param [in]  attributes      attributes to query for
# @param [in, out] data        a two-dimensional containing pointers to memory locations
# where the result of each attribute query will be written to
# @param [in]  ptr             pointer to get attributes for
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# @see hipPointerGetAttribute
cdef hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes,hipPointer_attribute * attributes,void ** data,hipDeviceptr_t ptr) nogil:
    global _lib_handle
    global hipDrvPointerGetAttributes_funptr
    if hipDrvPointerGetAttributes_funptr == NULL:
        with gil:
            hipDrvPointerGetAttributes_funptr = loader.load_symbol(_lib_handle, "hipDrvPointerGetAttributes")
    return (<hipError_t (*)(unsigned int,hipPointer_attribute *,void **,hipDeviceptr_t) nogil> hipDrvPointerGetAttributes_funptr)(numAttributes,attributes,data,ptr)


cdef void* hipImportExternalSemaphore_funptr = NULL
# @brief Imports an external semaphore.
# @param[out] extSem_out  External semaphores to be waited on
# @param[in] semHandleDesc Semaphore import handle descriptor
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @see
cdef hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,hipExternalSemaphoreHandleDesc_st * semHandleDesc) nogil:
    global _lib_handle
    global hipImportExternalSemaphore_funptr
    if hipImportExternalSemaphore_funptr == NULL:
        with gil:
            hipImportExternalSemaphore_funptr = loader.load_symbol(_lib_handle, "hipImportExternalSemaphore")
    return (<hipError_t (*)(hipExternalSemaphore_t*,hipExternalSemaphoreHandleDesc_st *) nogil> hipImportExternalSemaphore_funptr)(extSem_out,semHandleDesc)


cdef void* hipSignalExternalSemaphoresAsync_funptr = NULL
# @brief Signals a set of external semaphore objects.
# @param[in] extSem_out  External semaphores to be waited on
# @param[in] paramsArray Array of semaphore parameters
# @param[in] numExtSems Number of semaphores to wait on
# @param[in] stream Stream to enqueue the wait operations in
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @see
cdef hipError_t hipSignalExternalSemaphoresAsync(hipExternalSemaphore_t * extSemArray,hipExternalSemaphoreSignalParams_st * paramsArray,unsigned int numExtSems,hipStream_t stream) nogil:
    global _lib_handle
    global hipSignalExternalSemaphoresAsync_funptr
    if hipSignalExternalSemaphoresAsync_funptr == NULL:
        with gil:
            hipSignalExternalSemaphoresAsync_funptr = loader.load_symbol(_lib_handle, "hipSignalExternalSemaphoresAsync")
    return (<hipError_t (*)(hipExternalSemaphore_t *,hipExternalSemaphoreSignalParams_st *,unsigned int,hipStream_t) nogil> hipSignalExternalSemaphoresAsync_funptr)(extSemArray,paramsArray,numExtSems,stream)


cdef void* hipWaitExternalSemaphoresAsync_funptr = NULL
# @brief Waits on a set of external semaphore objects
# @param[in] extSem_out  External semaphores to be waited on
# @param[in] paramsArray Array of semaphore parameters
# @param[in] numExtSems Number of semaphores to wait on
# @param[in] stream Stream to enqueue the wait operations in
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @see
cdef hipError_t hipWaitExternalSemaphoresAsync(hipExternalSemaphore_t * extSemArray,hipExternalSemaphoreWaitParams_st * paramsArray,unsigned int numExtSems,hipStream_t stream) nogil:
    global _lib_handle
    global hipWaitExternalSemaphoresAsync_funptr
    if hipWaitExternalSemaphoresAsync_funptr == NULL:
        with gil:
            hipWaitExternalSemaphoresAsync_funptr = loader.load_symbol(_lib_handle, "hipWaitExternalSemaphoresAsync")
    return (<hipError_t (*)(hipExternalSemaphore_t *,hipExternalSemaphoreWaitParams_st *,unsigned int,hipStream_t) nogil> hipWaitExternalSemaphoresAsync_funptr)(extSemArray,paramsArray,numExtSems,stream)


cdef void* hipDestroyExternalSemaphore_funptr = NULL
# @brief Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.
# @param[in] extSem handle to an external memory object
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @see
cdef hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) nogil:
    global _lib_handle
    global hipDestroyExternalSemaphore_funptr
    if hipDestroyExternalSemaphore_funptr == NULL:
        with gil:
            hipDestroyExternalSemaphore_funptr = loader.load_symbol(_lib_handle, "hipDestroyExternalSemaphore")
    return (<hipError_t (*)(hipExternalSemaphore_t) nogil> hipDestroyExternalSemaphore_funptr)(extSem)


cdef void* hipImportExternalMemory_funptr = NULL
# @brief Imports an external memory object.
# @param[out] extMem_out  Returned handle to an external memory object
# @param[in]  memHandleDesc Memory import handle descriptor
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @see
cdef hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out,hipExternalMemoryHandleDesc_st * memHandleDesc) nogil:
    global _lib_handle
    global hipImportExternalMemory_funptr
    if hipImportExternalMemory_funptr == NULL:
        with gil:
            hipImportExternalMemory_funptr = loader.load_symbol(_lib_handle, "hipImportExternalMemory")
    return (<hipError_t (*)(hipExternalMemory_t*,hipExternalMemoryHandleDesc_st *) nogil> hipImportExternalMemory_funptr)(extMem_out,memHandleDesc)


cdef void* hipExternalMemoryGetMappedBuffer_funptr = NULL
# @brief Maps a buffer onto an imported memory object.
# @param[out] devPtr Returned device pointer to buffer
# @param[in]  extMem  Handle to external memory object
# @param[in]  bufferDesc  Buffer descriptor
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @see
cdef hipError_t hipExternalMemoryGetMappedBuffer(void ** devPtr,hipExternalMemory_t extMem,hipExternalMemoryBufferDesc_st * bufferDesc) nogil:
    global _lib_handle
    global hipExternalMemoryGetMappedBuffer_funptr
    if hipExternalMemoryGetMappedBuffer_funptr == NULL:
        with gil:
            hipExternalMemoryGetMappedBuffer_funptr = loader.load_symbol(_lib_handle, "hipExternalMemoryGetMappedBuffer")
    return (<hipError_t (*)(void **,hipExternalMemory_t,hipExternalMemoryBufferDesc_st *) nogil> hipExternalMemoryGetMappedBuffer_funptr)(devPtr,extMem,bufferDesc)


cdef void* hipDestroyExternalMemory_funptr = NULL
# @brief Destroys an external memory object.
# @param[in] extMem  External memory object to be destroyed
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @see
cdef hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) nogil:
    global _lib_handle
    global hipDestroyExternalMemory_funptr
    if hipDestroyExternalMemory_funptr == NULL:
        with gil:
            hipDestroyExternalMemory_funptr = loader.load_symbol(_lib_handle, "hipDestroyExternalMemory")
    return (<hipError_t (*)(hipExternalMemory_t) nogil> hipDestroyExternalMemory_funptr)(extMem)


cdef void* hipMalloc_funptr = NULL
# @brief Allocate memory on the default accelerator
# @param[out] ptr Pointer to the allocated memory
# @param[in]  size Requested memory size
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
# @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
# hipHostFree, hipHostMalloc
cdef hipError_t hipMalloc(void ** ptr,int size) nogil:
    global _lib_handle
    global hipMalloc_funptr
    if hipMalloc_funptr == NULL:
        with gil:
            hipMalloc_funptr = loader.load_symbol(_lib_handle, "hipMalloc")
    return (<hipError_t (*)(void **,int) nogil> hipMalloc_funptr)(ptr,size)


cdef void* hipExtMallocWithFlags_funptr = NULL
# @brief Allocate memory on the default accelerator
# @param[out] ptr Pointer to the allocated memory
# @param[in]  size Requested memory size
# @param[in]  flags Type of memory allocation
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
# @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
# hipHostFree, hipHostMalloc
cdef hipError_t hipExtMallocWithFlags(void ** ptr,int sizeBytes,unsigned int flags) nogil:
    global _lib_handle
    global hipExtMallocWithFlags_funptr
    if hipExtMallocWithFlags_funptr == NULL:
        with gil:
            hipExtMallocWithFlags_funptr = loader.load_symbol(_lib_handle, "hipExtMallocWithFlags")
    return (<hipError_t (*)(void **,int,unsigned int) nogil> hipExtMallocWithFlags_funptr)(ptr,sizeBytes,flags)


cdef void* hipMallocHost_funptr = NULL
# @brief Allocate pinned host memory [Deprecated]
# @param[out] ptr Pointer to the allocated host pinned memory
# @param[in]  size Requested memory size
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# @return #hipSuccess, #hipErrorOutOfMemory
# @deprecated use hipHostMalloc() instead
cdef hipError_t hipMallocHost(void ** ptr,int size) nogil:
    global _lib_handle
    global hipMallocHost_funptr
    if hipMallocHost_funptr == NULL:
        with gil:
            hipMallocHost_funptr = loader.load_symbol(_lib_handle, "hipMallocHost")
    return (<hipError_t (*)(void **,int) nogil> hipMallocHost_funptr)(ptr,size)


cdef void* hipMemAllocHost_funptr = NULL
# @brief Allocate pinned host memory [Deprecated]
# @param[out] ptr Pointer to the allocated host pinned memory
# @param[in]  size Requested memory size
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# @return #hipSuccess, #hipErrorOutOfMemory
# @deprecated use hipHostMalloc() instead
cdef hipError_t hipMemAllocHost(void ** ptr,int size) nogil:
    global _lib_handle
    global hipMemAllocHost_funptr
    if hipMemAllocHost_funptr == NULL:
        with gil:
            hipMemAllocHost_funptr = loader.load_symbol(_lib_handle, "hipMemAllocHost")
    return (<hipError_t (*)(void **,int) nogil> hipMemAllocHost_funptr)(ptr,size)


cdef void* hipHostMalloc_funptr = NULL
# @brief Allocate device accessible page locked host memory
# @param[out] ptr Pointer to the allocated host pinned memory
# @param[in]  size Requested memory size
# @param[in]  flags Type of host memory allocation
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# @return #hipSuccess, #hipErrorOutOfMemory
# @see hipSetDeviceFlags, hipHostFree
cdef hipError_t hipHostMalloc(void ** ptr,int size,unsigned int flags) nogil:
    global _lib_handle
    global hipHostMalloc_funptr
    if hipHostMalloc_funptr == NULL:
        with gil:
            hipHostMalloc_funptr = loader.load_symbol(_lib_handle, "hipHostMalloc")
    return (<hipError_t (*)(void **,int,unsigned int) nogil> hipHostMalloc_funptr)(ptr,size,flags)


cdef void* hipMallocManaged_funptr = NULL
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# @addtogroup MemoryM Managed Memory
# @{
# @ingroup Memory
# This section describes the managed memory management functions of HIP runtime API.
# @brief Allocates memory that will be automatically managed by HIP.
# @param [out] dev_ptr - pointer to allocated device memory
# @param [in]  size    - requested allocation size in bytes
# @param [in]  flags   - must be either hipMemAttachGlobal or hipMemAttachHost
# (defaults to hipMemAttachGlobal)
# @returns #hipSuccess, #hipErrorMemoryAllocation, #hipErrorNotSupported, #hipErrorInvalidValue
cdef hipError_t hipMallocManaged(void ** dev_ptr,int size,unsigned int flags) nogil:
    global _lib_handle
    global hipMallocManaged_funptr
    if hipMallocManaged_funptr == NULL:
        with gil:
            hipMallocManaged_funptr = loader.load_symbol(_lib_handle, "hipMallocManaged")
    return (<hipError_t (*)(void **,int,unsigned int) nogil> hipMallocManaged_funptr)(dev_ptr,size,flags)


cdef void* hipMemPrefetchAsync_funptr = NULL
# @brief Prefetches memory to the specified destination device using HIP.
# @param [in] dev_ptr  pointer to be prefetched
# @param [in] count    size in bytes for prefetching
# @param [in] device   destination device to prefetch to
# @param [in] stream   stream to enqueue prefetch operation
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemPrefetchAsync(const void * dev_ptr,int count,int device,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemPrefetchAsync_funptr
    if hipMemPrefetchAsync_funptr == NULL:
        with gil:
            hipMemPrefetchAsync_funptr = loader.load_symbol(_lib_handle, "hipMemPrefetchAsync")
    return (<hipError_t (*)(const void *,int,int,hipStream_t) nogil> hipMemPrefetchAsync_funptr)(dev_ptr,count,device,stream)


cdef void* hipMemAdvise_funptr = NULL
# @brief Advise about the usage of a given memory range to HIP.
# @param [in] dev_ptr  pointer to memory to set the advice for
# @param [in] count    size in bytes of the memory range
# @param [in] advice   advice to be applied for the specified memory range
# @param [in] device   device to apply the advice for
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemAdvise(const void * dev_ptr,int count,hipMemoryAdvise advice,int device) nogil:
    global _lib_handle
    global hipMemAdvise_funptr
    if hipMemAdvise_funptr == NULL:
        with gil:
            hipMemAdvise_funptr = loader.load_symbol(_lib_handle, "hipMemAdvise")
    return (<hipError_t (*)(const void *,int,hipMemoryAdvise,int) nogil> hipMemAdvise_funptr)(dev_ptr,count,advice,device)


cdef void* hipMemRangeGetAttribute_funptr = NULL
# @brief Query an attribute of a given memory range in HIP.
# @param [in,out] data   a pointer to a memory location where the result of each
# attribute query will be written to
# @param [in] data_size  the size of data
# @param [in] attribute  the attribute to query
# @param [in] dev_ptr    start of the range to query
# @param [in] count      size of the range to query
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemRangeGetAttribute(void * data,int data_size,hipMemRangeAttribute attribute,const void * dev_ptr,int count) nogil:
    global _lib_handle
    global hipMemRangeGetAttribute_funptr
    if hipMemRangeGetAttribute_funptr == NULL:
        with gil:
            hipMemRangeGetAttribute_funptr = loader.load_symbol(_lib_handle, "hipMemRangeGetAttribute")
    return (<hipError_t (*)(void *,int,hipMemRangeAttribute,const void *,int) nogil> hipMemRangeGetAttribute_funptr)(data,data_size,attribute,dev_ptr,count)


cdef void* hipMemRangeGetAttributes_funptr = NULL
# @brief Query attributes of a given memory range in HIP.
# @param [in,out] data     a two-dimensional array containing pointers to memory locations
# where the result of each attribute query will be written to
# @param [in] data_sizes   an array, containing the sizes of each result
# @param [in] attributes   the attribute to query
# @param [in] num_attributes  an array of attributes to query (numAttributes and the number
# of attributes in this array should match)
# @param [in] dev_ptr      start of the range to query
# @param [in] count        size of the range to query
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemRangeGetAttributes(void ** data,int * data_sizes,hipMemRangeAttribute * attributes,int num_attributes,const void * dev_ptr,int count) nogil:
    global _lib_handle
    global hipMemRangeGetAttributes_funptr
    if hipMemRangeGetAttributes_funptr == NULL:
        with gil:
            hipMemRangeGetAttributes_funptr = loader.load_symbol(_lib_handle, "hipMemRangeGetAttributes")
    return (<hipError_t (*)(void **,int *,hipMemRangeAttribute *,int,const void *,int) nogil> hipMemRangeGetAttributes_funptr)(data,data_sizes,attributes,num_attributes,dev_ptr,count)


cdef void* hipStreamAttachMemAsync_funptr = NULL
# @brief Attach memory to a stream asynchronously in HIP.
# @param [in] stream     - stream in which to enqueue the attach operation
# @param [in] dev_ptr    - pointer to memory (must be a pointer to managed memory or
# to a valid host-accessible region of system-allocated memory)
# @param [in] length     - length of memory (defaults to zero)
# @param [in] flags      - must be one of hipMemAttachGlobal, hipMemAttachHost or
# hipMemAttachSingle (defaults to hipMemAttachSingle)
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipStreamAttachMemAsync(hipStream_t stream,void * dev_ptr,int length,unsigned int flags) nogil:
    global _lib_handle
    global hipStreamAttachMemAsync_funptr
    if hipStreamAttachMemAsync_funptr == NULL:
        with gil:
            hipStreamAttachMemAsync_funptr = loader.load_symbol(_lib_handle, "hipStreamAttachMemAsync")
    return (<hipError_t (*)(hipStream_t,void *,int,unsigned int) nogil> hipStreamAttachMemAsync_funptr)(stream,dev_ptr,length,flags)


cdef void* hipMallocAsync_funptr = NULL
# @brief Allocates memory with stream ordered semantics
# Inserts a memory allocation operation into @p stream.
# A pointer to the allocated memory is returned immediately in *dptr.
# The allocation must not be accessed until the the allocation operation completes.
# The allocation comes from the memory pool associated with the stream's device.
# @note The default memory pool of a device contains device memory from that device.
# @note Basic stream ordering allows future work submitted into the same stream to use the allocation.
# Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
# operation completes before work submitted in a separate stream runs.
# @note During stream capture, this function results in the creation of an allocation node. In this case,
# the allocation is owned by the graph instead of the memory pool. The memory pool's properties
# are used to set the node's creation parameters.
# @param [out] dev_ptr  Returned device pointer of memory allocation
# @param [in] size      Number of bytes to allocate
# @param [in] stream    The stream establishing the stream ordering contract and
# the memory pool to allocate from
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
# @see hipMallocFromPoolAsync, hipFreeAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMallocAsync(void ** dev_ptr,int size,hipStream_t stream) nogil:
    global _lib_handle
    global hipMallocAsync_funptr
    if hipMallocAsync_funptr == NULL:
        with gil:
            hipMallocAsync_funptr = loader.load_symbol(_lib_handle, "hipMallocAsync")
    return (<hipError_t (*)(void **,int,hipStream_t) nogil> hipMallocAsync_funptr)(dev_ptr,size,stream)


cdef void* hipFreeAsync_funptr = NULL
# @brief Frees memory with stream ordered semantics
# Inserts a free operation into @p stream.
# The allocation must not be used after stream execution reaches the free.
# After this API returns, accessing the memory from any subsequent work launched on the GPU
# or querying its pointer attributes results in undefined behavior.
# @note During stream capture, this function results in the creation of a free node and
# must therefore be passed the address of a graph allocation.
# @param [in] dev_ptr Pointer to device memory to free
# @param [in] stream  The stream, where the destruciton will occur according to the execution order
# @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported
# @see hipMallocFromPoolAsync, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipFreeAsync(void * dev_ptr,hipStream_t stream) nogil:
    global _lib_handle
    global hipFreeAsync_funptr
    if hipFreeAsync_funptr == NULL:
        with gil:
            hipFreeAsync_funptr = loader.load_symbol(_lib_handle, "hipFreeAsync")
    return (<hipError_t (*)(void *,hipStream_t) nogil> hipFreeAsync_funptr)(dev_ptr,stream)


cdef void* hipMemPoolTrimTo_funptr = NULL
# @brief Releases freed memory back to the OS
# Releases memory back to the OS until the pool contains fewer than @p min_bytes_to_keep
# reserved bytes, or there is no more memory that the allocator can safely release.
# The allocator cannot release OS allocations that back outstanding asynchronous allocations.
# The OS allocations may happen at different granularity from the user allocations.
# @note: Allocations that have not been freed count as outstanding.
# @note: Allocations that have been asynchronously freed but whose completion has
# not been observed on the host (eg. by a synchronize) can count as outstanding.
# @param[in] mem_pool          The memory pool to trim allocations
# @param[in] min_bytes_to_hold If the pool has less than min_bytes_to_hold reserved,
# then the TrimTo operation is a no-op.  Otherwise the memory pool will contain
# at least min_bytes_to_hold bytes reserved after the operation.
# @returns #hipSuccess, #hipErrorInvalidValue
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool,int min_bytes_to_hold) nogil:
    global _lib_handle
    global hipMemPoolTrimTo_funptr
    if hipMemPoolTrimTo_funptr == NULL:
        with gil:
            hipMemPoolTrimTo_funptr = loader.load_symbol(_lib_handle, "hipMemPoolTrimTo")
    return (<hipError_t (*)(hipMemPool_t,int) nogil> hipMemPoolTrimTo_funptr)(mem_pool,min_bytes_to_hold)


cdef void* hipMemPoolSetAttribute_funptr = NULL
# @brief Sets attributes of a memory pool
# Supported attributes are:
# - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
# Amount of reserved memory in bytes to hold onto before trying
# to release memory back to the OS. When more than the release
# threshold bytes of memory are held by the memory pool, the
# allocator will try to release memory back to the OS on the
# next call to stream, event or context synchronize. (default 0)
# - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
# Allow @p hipMallocAsync to use memory asynchronously freed
# in another stream as long as a stream ordering dependency
# of the allocating stream on the free action exists.
# HIP events and null stream interactions can create the required
# stream ordered dependencies. (default enabled)
# - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
# Allow reuse of already completed frees when there is no dependency
# between the free and allocation. (default enabled)
# - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
# Allow @p hipMallocAsync to insert new stream dependencies
# in order to establish the stream ordering required to reuse
# a piece of memory released by @p hipFreeAsync (default enabled).
# @param [in] mem_pool The memory pool to modify
# @param [in] attr     The attribute to modify
# @param [in] value    Pointer to the value to assign
# @returns #hipSuccess, #hipErrorInvalidValue
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool,hipMemPoolAttr attr,void * value) nogil:
    global _lib_handle
    global hipMemPoolSetAttribute_funptr
    if hipMemPoolSetAttribute_funptr == NULL:
        with gil:
            hipMemPoolSetAttribute_funptr = loader.load_symbol(_lib_handle, "hipMemPoolSetAttribute")
    return (<hipError_t (*)(hipMemPool_t,hipMemPoolAttr,void *) nogil> hipMemPoolSetAttribute_funptr)(mem_pool,attr,value)


cdef void* hipMemPoolGetAttribute_funptr = NULL
# @brief Gets attributes of a memory pool
# Supported attributes are:
# - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
# Amount of reserved memory in bytes to hold onto before trying
# to release memory back to the OS. When more than the release
# threshold bytes of memory are held by the memory pool, the
# allocator will try to release memory back to the OS on the
# next call to stream, event or context synchronize. (default 0)
# - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
# Allow @p hipMallocAsync to use memory asynchronously freed
# in another stream as long as a stream ordering dependency
# of the allocating stream on the free action exists.
# HIP events and null stream interactions can create the required
# stream ordered dependencies. (default enabled)
# - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
# Allow reuse of already completed frees when there is no dependency
# between the free and allocation. (default enabled)
# - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
# Allow @p hipMallocAsync to insert new stream dependencies
# in order to establish the stream ordering required to reuse
# a piece of memory released by @p hipFreeAsync (default enabled).
# @param [in] mem_pool The memory pool to get attributes of
# @param [in] attr     The attribute to get
# @param [in] value    Retrieved value
# @returns  #hipSuccess, #hipErrorInvalidValue
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool,hipMemPoolAttr attr,void * value) nogil:
    global _lib_handle
    global hipMemPoolGetAttribute_funptr
    if hipMemPoolGetAttribute_funptr == NULL:
        with gil:
            hipMemPoolGetAttribute_funptr = loader.load_symbol(_lib_handle, "hipMemPoolGetAttribute")
    return (<hipError_t (*)(hipMemPool_t,hipMemPoolAttr,void *) nogil> hipMemPoolGetAttribute_funptr)(mem_pool,attr,value)


cdef void* hipMemPoolSetAccess_funptr = NULL
# @brief Controls visibility of the specified pool between devices
# @param [in] mem_pool   Memory pool for acccess change
# @param [in] desc_list  Array of access descriptors. Each descriptor instructs the access to enable for a single gpu
# @param [in] count  Number of descriptors in the map array.
# @returns  #hipSuccess, #hipErrorInvalidValue
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool,hipMemAccessDesc * desc_list,int count) nogil:
    global _lib_handle
    global hipMemPoolSetAccess_funptr
    if hipMemPoolSetAccess_funptr == NULL:
        with gil:
            hipMemPoolSetAccess_funptr = loader.load_symbol(_lib_handle, "hipMemPoolSetAccess")
    return (<hipError_t (*)(hipMemPool_t,hipMemAccessDesc *,int) nogil> hipMemPoolSetAccess_funptr)(mem_pool,desc_list,count)


cdef void* hipMemPoolGetAccess_funptr = NULL
# @brief Returns the accessibility of a pool from a device
# Returns the accessibility of the pool's memory from the specified location.
# @param [out] flags    Accessibility of the memory pool from the specified location/device
# @param [in] mem_pool   Memory pool being queried
# @param [in] location  Location/device for memory pool access
# @returns #hipSuccess, #hipErrorInvalidValue
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolGetAccess(hipMemAccessFlags * flags,hipMemPool_t mem_pool,hipMemLocation * location) nogil:
    global _lib_handle
    global hipMemPoolGetAccess_funptr
    if hipMemPoolGetAccess_funptr == NULL:
        with gil:
            hipMemPoolGetAccess_funptr = loader.load_symbol(_lib_handle, "hipMemPoolGetAccess")
    return (<hipError_t (*)(hipMemAccessFlags *,hipMemPool_t,hipMemLocation *) nogil> hipMemPoolGetAccess_funptr)(flags,mem_pool,location)


cdef void* hipMemPoolCreate_funptr = NULL
# @brief Creates a memory pool
# Creates a HIP memory pool and returns the handle in @p mem_pool. The @p pool_props determines
# the properties of the pool such as the backing device and IPC capabilities.
# By default, the memory pool will be accessible from the device it is allocated on.
# @param [out] mem_pool    Contains createed memory pool
# @param [in] pool_props   Memory pool properties
# @note Specifying hipMemHandleTypeNone creates a memory pool that will not support IPC.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolDestroy,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool,hipMemPoolProps * pool_props) nogil:
    global _lib_handle
    global hipMemPoolCreate_funptr
    if hipMemPoolCreate_funptr == NULL:
        with gil:
            hipMemPoolCreate_funptr = loader.load_symbol(_lib_handle, "hipMemPoolCreate")
    return (<hipError_t (*)(hipMemPool_t*,hipMemPoolProps *) nogil> hipMemPoolCreate_funptr)(mem_pool,pool_props)


cdef void* hipMemPoolDestroy_funptr = NULL
# @brief Destroys the specified memory pool
# If any pointers obtained from this pool haven't been freed or
# the pool has free operations that haven't completed
# when @p hipMemPoolDestroy is invoked, the function will return immediately and the
# resources associated with the pool will be released automatically
# once there are no more outstanding allocations.
# Destroying the current mempool of a device sets the default mempool of
# that device as the current mempool for that device.
# @param [in] mem_pool Memory pool for destruction
# @note A device's default memory pool cannot be destroyed.
# @returns #hipSuccess, #hipErrorInvalidValue
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) nogil:
    global _lib_handle
    global hipMemPoolDestroy_funptr
    if hipMemPoolDestroy_funptr == NULL:
        with gil:
            hipMemPoolDestroy_funptr = loader.load_symbol(_lib_handle, "hipMemPoolDestroy")
    return (<hipError_t (*)(hipMemPool_t) nogil> hipMemPoolDestroy_funptr)(mem_pool)


cdef void* hipMallocFromPoolAsync_funptr = NULL
# @brief Allocates memory from a specified pool with stream ordered semantics.
# Inserts an allocation operation into @p stream.
# A pointer to the allocated memory is returned immediately in @p dev_ptr.
# The allocation must not be accessed until the the allocation operation completes.
# The allocation comes from the specified memory pool.
# @note The specified memory pool may be from a device different than that of the specified @p stream.
# Basic stream ordering allows future work submitted into the same stream to use the allocation.
# Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
# operation completes before work submitted in a separate stream runs.
# @note During stream capture, this function results in the creation of an allocation node. In this case,
# the allocation is owned by the graph instead of the memory pool. The memory pool's properties
# are used to set the node's creation parameters.
# @param [out] dev_ptr Returned device pointer
# @param [in] size     Number of bytes to allocate
# @param [in] mem_pool The pool to allocate from
# @param [in] stream   The stream establishing the stream ordering semantic
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
# @see hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess,
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMallocFromPoolAsync(void ** dev_ptr,int size,hipMemPool_t mem_pool,hipStream_t stream) nogil:
    global _lib_handle
    global hipMallocFromPoolAsync_funptr
    if hipMallocFromPoolAsync_funptr == NULL:
        with gil:
            hipMallocFromPoolAsync_funptr = loader.load_symbol(_lib_handle, "hipMallocFromPoolAsync")
    return (<hipError_t (*)(void **,int,hipMemPool_t,hipStream_t) nogil> hipMallocFromPoolAsync_funptr)(dev_ptr,size,mem_pool,stream)


cdef void* hipMemPoolExportToShareableHandle_funptr = NULL
# @brief Exports a memory pool to the requested handle type.
# Given an IPC capable mempool, create an OS handle to share the pool with another process.
# A recipient process can convert the shareable handle into a mempool with @p hipMemPoolImportFromShareableHandle.
# Individual pointers can then be shared with the @p hipMemPoolExportPointer and @p hipMemPoolImportPointer APIs.
# The implementation of what the shareable handle is and how it can be transferred is defined by the requested
# handle type.
# @note: To create an IPC capable mempool, create a mempool with a @p hipMemAllocationHandleType other
# than @p hipMemHandleTypeNone.
# @param [out] shared_handle Pointer to the location in which to store the requested handle
# @param [in] mem_pool       Pool to export
# @param [in] handle_type    The type of handle to create
# @param [in] flags          Must be 0
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
# @see hipMemPoolImportFromShareableHandle
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolExportToShareableHandle(void * shared_handle,hipMemPool_t mem_pool,hipMemAllocationHandleType handle_type,unsigned int flags) nogil:
    global _lib_handle
    global hipMemPoolExportToShareableHandle_funptr
    if hipMemPoolExportToShareableHandle_funptr == NULL:
        with gil:
            hipMemPoolExportToShareableHandle_funptr = loader.load_symbol(_lib_handle, "hipMemPoolExportToShareableHandle")
    return (<hipError_t (*)(void *,hipMemPool_t,hipMemAllocationHandleType,unsigned int) nogil> hipMemPoolExportToShareableHandle_funptr)(shared_handle,mem_pool,handle_type,flags)


cdef void* hipMemPoolImportFromShareableHandle_funptr = NULL
# @brief Imports a memory pool from a shared handle.
# Specific allocations can be imported from the imported pool with @p hipMemPoolImportPointer.
# @note Imported memory pools do not support creating new allocations.
# As such imported memory pools may not be used in @p hipDeviceSetMemPool
# or @p hipMallocFromPoolAsync calls.
# @param [out] mem_pool     Returned memory pool
# @param [in] shared_handle OS handle of the pool to open
# @param [in] handle_type   The type of handle being imported
# @param [in] flags         Must be 0
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
# @see hipMemPoolExportToShareableHandle
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool,void * shared_handle,hipMemAllocationHandleType handle_type,unsigned int flags) nogil:
    global _lib_handle
    global hipMemPoolImportFromShareableHandle_funptr
    if hipMemPoolImportFromShareableHandle_funptr == NULL:
        with gil:
            hipMemPoolImportFromShareableHandle_funptr = loader.load_symbol(_lib_handle, "hipMemPoolImportFromShareableHandle")
    return (<hipError_t (*)(hipMemPool_t*,void *,hipMemAllocationHandleType,unsigned int) nogil> hipMemPoolImportFromShareableHandle_funptr)(mem_pool,shared_handle,handle_type,flags)


cdef void* hipMemPoolExportPointer_funptr = NULL
# @brief Export data to share a memory pool allocation between processes.
# Constructs @p export_data for sharing a specific allocation from an already shared memory pool.
# The recipient process can import the allocation with the @p hipMemPoolImportPointer api.
# The data is not a handle and may be shared through any IPC mechanism.
# @param[out] export_data  Returned export data
# @param[in] dev_ptr       Pointer to memory being exported
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
# @see hipMemPoolImportPointer
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData * export_data,void * dev_ptr) nogil:
    global _lib_handle
    global hipMemPoolExportPointer_funptr
    if hipMemPoolExportPointer_funptr == NULL:
        with gil:
            hipMemPoolExportPointer_funptr = loader.load_symbol(_lib_handle, "hipMemPoolExportPointer")
    return (<hipError_t (*)(hipMemPoolPtrExportData *,void *) nogil> hipMemPoolExportPointer_funptr)(export_data,dev_ptr)


cdef void* hipMemPoolImportPointer_funptr = NULL
# @brief Import a memory pool allocation from another process.
# Returns in @p dev_ptr a pointer to the imported memory.
# The imported memory must not be accessed before the allocation operation completes
# in the exporting process. The imported memory must be freed from all importing processes before
# being freed in the exporting process. The pointer may be freed with @p hipFree
# or @p hipFreeAsync. If @p hipFreeAsync is used, the free must be completed
# on the importing process before the free operation on the exporting process.
# @note The @p hipFreeAsync api may be used in the exporting process before
# the @p hipFreeAsync operation completes in its stream as long as the
# @p hipFreeAsync in the exporting process specifies a stream with
# a stream dependency on the importing process's @p hipFreeAsync.
# @param [out] dev_ptr     Pointer to imported memory
# @param [in] mem_pool     Memory pool from which to import a pointer
# @param [in] export_data  Data specifying the memory to import
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized, #hipErrorOutOfMemory
# @see hipMemPoolExportPointer
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolImportPointer(void ** dev_ptr,hipMemPool_t mem_pool,hipMemPoolPtrExportData * export_data) nogil:
    global _lib_handle
    global hipMemPoolImportPointer_funptr
    if hipMemPoolImportPointer_funptr == NULL:
        with gil:
            hipMemPoolImportPointer_funptr = loader.load_symbol(_lib_handle, "hipMemPoolImportPointer")
    return (<hipError_t (*)(void **,hipMemPool_t,hipMemPoolPtrExportData *) nogil> hipMemPoolImportPointer_funptr)(dev_ptr,mem_pool,export_data)


cdef void* hipHostAlloc_funptr = NULL
# @brief Allocate device accessible page locked host memory [Deprecated]
# @param[out] ptr Pointer to the allocated host pinned memory
# @param[in]  size Requested memory size
# @param[in]  flags Type of host memory allocation
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# @return #hipSuccess, #hipErrorOutOfMemory
# @deprecated use hipHostMalloc() instead
cdef hipError_t hipHostAlloc(void ** ptr,int size,unsigned int flags) nogil:
    global _lib_handle
    global hipHostAlloc_funptr
    if hipHostAlloc_funptr == NULL:
        with gil:
            hipHostAlloc_funptr = loader.load_symbol(_lib_handle, "hipHostAlloc")
    return (<hipError_t (*)(void **,int,unsigned int) nogil> hipHostAlloc_funptr)(ptr,size,flags)


cdef void* hipHostGetDevicePointer_funptr = NULL
# @brief Get Device pointer from Host Pointer allocated through hipHostMalloc
# @param[out] dstPtr Device Pointer mapped to passed host pointer
# @param[in]  hstPtr Host Pointer allocated through hipHostMalloc
# @param[in]  flags Flags to be passed for extension
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
# @see hipSetDeviceFlags, hipHostMalloc
cdef hipError_t hipHostGetDevicePointer(void ** devPtr,void * hstPtr,unsigned int flags) nogil:
    global _lib_handle
    global hipHostGetDevicePointer_funptr
    if hipHostGetDevicePointer_funptr == NULL:
        with gil:
            hipHostGetDevicePointer_funptr = loader.load_symbol(_lib_handle, "hipHostGetDevicePointer")
    return (<hipError_t (*)(void **,void *,unsigned int) nogil> hipHostGetDevicePointer_funptr)(devPtr,hstPtr,flags)


cdef void* hipHostGetFlags_funptr = NULL
# @brief Return flags associated with host pointer
# @param[out] flagsPtr Memory location to store flags
# @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
# @return #hipSuccess, #hipErrorInvalidValue
# @see hipHostMalloc
cdef hipError_t hipHostGetFlags(unsigned int * flagsPtr,void * hostPtr) nogil:
    global _lib_handle
    global hipHostGetFlags_funptr
    if hipHostGetFlags_funptr == NULL:
        with gil:
            hipHostGetFlags_funptr = loader.load_symbol(_lib_handle, "hipHostGetFlags")
    return (<hipError_t (*)(unsigned int *,void *) nogil> hipHostGetFlags_funptr)(flagsPtr,hostPtr)


cdef void* hipHostRegister_funptr = NULL
# @brief Register host memory so it can be accessed from the current device.
# @param[out] hostPtr Pointer to host memory to be registered.
# @param[in] sizeBytes size of the host memory
# @param[in] flags.  See below.
# Flags:
# - #hipHostRegisterDefault   Memory is Mapped and Portable
# - #hipHostRegisterPortable  Memory is considered registered by all contexts.  HIP only supports
# one context so this is always assumed true.
# - #hipHostRegisterMapped    Map the allocation into the address space for the current device.
# The device pointer can be obtained with #hipHostGetDevicePointer.
# After registering the memory, use #hipHostGetDevicePointer to obtain the mapped device pointer.
# On many systems, the mapped device pointer will have a different value than the mapped host
# pointer.  Applications must use the device pointer in device code, and the host pointer in device
# code.
# On some systems, registered memory is pinned.  On some systems, registered memory may not be
# actually be pinned but uses OS or hardware facilities to all GPU access to the host memory.
# Developers are strongly encouraged to register memory blocks which are aligned to the host
# cache-line size. (typically 64-bytes but can be obtains from the CPUID instruction).
# If registering non-aligned pointers, the application must take care when register pointers from
# the same cache line on different devices.  HIP's coarse-grained synchronization model does not
# guarantee correct results if different devices write to different parts of the same cache block -
# typically one of the writes will "win" and overwrite data from the other registered memory
# region.
# @return #hipSuccess, #hipErrorOutOfMemory
# @see hipHostUnregister, hipHostGetFlags, hipHostGetDevicePointer
cdef hipError_t hipHostRegister(void * hostPtr,int sizeBytes,unsigned int flags) nogil:
    global _lib_handle
    global hipHostRegister_funptr
    if hipHostRegister_funptr == NULL:
        with gil:
            hipHostRegister_funptr = loader.load_symbol(_lib_handle, "hipHostRegister")
    return (<hipError_t (*)(void *,int,unsigned int) nogil> hipHostRegister_funptr)(hostPtr,sizeBytes,flags)


cdef void* hipHostUnregister_funptr = NULL
# @brief Un-register host pointer
# @param[in] hostPtr Host pointer previously registered with #hipHostRegister
# @return Error code
# @see hipHostRegister
cdef hipError_t hipHostUnregister(void * hostPtr) nogil:
    global _lib_handle
    global hipHostUnregister_funptr
    if hipHostUnregister_funptr == NULL:
        with gil:
            hipHostUnregister_funptr = loader.load_symbol(_lib_handle, "hipHostUnregister")
    return (<hipError_t (*)(void *) nogil> hipHostUnregister_funptr)(hostPtr)


cdef void* hipMallocPitch_funptr = NULL
# Allocates at least width (in bytes) * height bytes of linear memory
# Padding may occur to ensure alighnment requirements are met for the given row
# The change in width size due to padding will be returned in *pitch.
# Currently the alignment is set to 128 bytes
# @param[out] ptr Pointer to the allocated device memory
# @param[out] pitch Pitch for allocation (in bytes)
# @param[in]  width Requested pitched allocation width (in bytes)
# @param[in]  height Requested pitched allocation height
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# @return Error code
# @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
# hipMalloc3DArray, hipHostMalloc
cdef hipError_t hipMallocPitch(void ** ptr,int * pitch,int width,int height) nogil:
    global _lib_handle
    global hipMallocPitch_funptr
    if hipMallocPitch_funptr == NULL:
        with gil:
            hipMallocPitch_funptr = loader.load_symbol(_lib_handle, "hipMallocPitch")
    return (<hipError_t (*)(void **,int *,int,int) nogil> hipMallocPitch_funptr)(ptr,pitch,width,height)


cdef void* hipMemAllocPitch_funptr = NULL
# Allocates at least width (in bytes) * height bytes of linear memory
# Padding may occur to ensure alighnment requirements are met for the given row
# The change in width size due to padding will be returned in *pitch.
# Currently the alignment is set to 128 bytes
# @param[out] dptr Pointer to the allocated device memory
# @param[out] pitch Pitch for allocation (in bytes)
# @param[in]  width Requested pitched allocation width (in bytes)
# @param[in]  height Requested pitched allocation height
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array.
# Given the row and column of an array element of type T, the address is computed as:
# T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
# @return Error code
# @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
# hipMalloc3DArray, hipHostMalloc
cdef hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr,int * pitch,int widthInBytes,int height,unsigned int elementSizeBytes) nogil:
    global _lib_handle
    global hipMemAllocPitch_funptr
    if hipMemAllocPitch_funptr == NULL:
        with gil:
            hipMemAllocPitch_funptr = loader.load_symbol(_lib_handle, "hipMemAllocPitch")
    return (<hipError_t (*)(hipDeviceptr_t*,int *,int,int,unsigned int) nogil> hipMemAllocPitch_funptr)(dptr,pitch,widthInBytes,height,elementSizeBytes)


cdef void* hipFree_funptr = NULL
# @brief Free memory allocated by the hcc hip memory allocation API.
# This API performs an implicit hipDeviceSynchronize() call.
# If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
# @param[in] ptr Pointer to memory to be freed
# @return #hipSuccess
# @return #hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated
# with hipHostMalloc)
# @see hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
# hipMalloc3DArray, hipHostMalloc
cdef hipError_t hipFree(void * ptr) nogil:
    global _lib_handle
    global hipFree_funptr
    if hipFree_funptr == NULL:
        with gil:
            hipFree_funptr = loader.load_symbol(_lib_handle, "hipFree")
    return (<hipError_t (*)(void *) nogil> hipFree_funptr)(ptr)


cdef void* hipFreeHost_funptr = NULL
# @brief Free memory allocated by the hcc hip host memory allocation API.  [Deprecated]
# @param[in] ptr Pointer to memory to be freed
# @return #hipSuccess,
# #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
#  hipMalloc)
# @deprecated use hipHostFree() instead
cdef hipError_t hipFreeHost(void * ptr) nogil:
    global _lib_handle
    global hipFreeHost_funptr
    if hipFreeHost_funptr == NULL:
        with gil:
            hipFreeHost_funptr = loader.load_symbol(_lib_handle, "hipFreeHost")
    return (<hipError_t (*)(void *) nogil> hipFreeHost_funptr)(ptr)


cdef void* hipHostFree_funptr = NULL
# @brief Free memory allocated by the hcc hip host memory allocation API
# This API performs an implicit hipDeviceSynchronize() call.
# If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
# @param[in] ptr Pointer to memory to be freed
# @return #hipSuccess,
# #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
# hipMalloc)
# @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D,
# hipMalloc3DArray, hipHostMalloc
cdef hipError_t hipHostFree(void * ptr) nogil:
    global _lib_handle
    global hipHostFree_funptr
    if hipHostFree_funptr == NULL:
        with gil:
            hipHostFree_funptr = loader.load_symbol(_lib_handle, "hipHostFree")
    return (<hipError_t (*)(void *) nogil> hipHostFree_funptr)(ptr)


cdef void* hipMemcpy_funptr = NULL
# @brief Copy data from src to dst.
# It supports memory from host to device,
# device to host, device to device and host to host
# The src and dst must not overlap.
# For hipMemcpy, the copy is always performed by the current device (set by hipSetDevice).
# For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the
# device where the src data is physically located. For optimal peer-to-peer copies, the copy device
# must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy
# agent as the current device and src/dest as the peerDevice argument.  if this is not done, the
# hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
# Calling hipMemcpy with dst and src pointers that do not match the hipMemcpyKind results in
# undefined behavior.
# @param[out]  dst Data being copy to
# @param[in]  src Data being copy from
# @param[in]  sizeBytes Data size in bytes
# @param[in]  copyType Memory copy type
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknowni
# @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpy(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpy_funptr
    if hipMemcpy_funptr == NULL:
        with gil:
            hipMemcpy_funptr = loader.load_symbol(_lib_handle, "hipMemcpy")
    return (<hipError_t (*)(void *,const void *,int,hipMemcpyKind) nogil> hipMemcpy_funptr)(dst,src,sizeBytes,kind)


cdef void* hipMemcpyWithStream_funptr = NULL
cdef hipError_t hipMemcpyWithStream(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyWithStream_funptr
    if hipMemcpyWithStream_funptr == NULL:
        with gil:
            hipMemcpyWithStream_funptr = loader.load_symbol(_lib_handle, "hipMemcpyWithStream")
    return (<hipError_t (*)(void *,const void *,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpyWithStream_funptr)(dst,src,sizeBytes,kind,stream)


cdef void* hipMemcpyHtoD_funptr = NULL
# @brief Copy data from Host to Device
# @param[out]  dst Data being copy to
# @param[in]   src Data being copy from
# @param[in]   sizeBytes Data size in bytes
# @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyHtoD(hipDeviceptr_t dst,void * src,int sizeBytes) nogil:
    global _lib_handle
    global hipMemcpyHtoD_funptr
    if hipMemcpyHtoD_funptr == NULL:
        with gil:
            hipMemcpyHtoD_funptr = loader.load_symbol(_lib_handle, "hipMemcpyHtoD")
    return (<hipError_t (*)(hipDeviceptr_t,void *,int) nogil> hipMemcpyHtoD_funptr)(dst,src,sizeBytes)


cdef void* hipMemcpyDtoH_funptr = NULL
# @brief Copy data from Device to Host
# @param[out]  dst Data being copy to
# @param[in]   src Data being copy from
# @param[in]   sizeBytes Data size in bytes
# @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyDtoH(void * dst,hipDeviceptr_t src,int sizeBytes) nogil:
    global _lib_handle
    global hipMemcpyDtoH_funptr
    if hipMemcpyDtoH_funptr == NULL:
        with gil:
            hipMemcpyDtoH_funptr = loader.load_symbol(_lib_handle, "hipMemcpyDtoH")
    return (<hipError_t (*)(void *,hipDeviceptr_t,int) nogil> hipMemcpyDtoH_funptr)(dst,src,sizeBytes)


cdef void* hipMemcpyDtoD_funptr = NULL
# @brief Copy data from Device to Device
# @param[out]  dst Data being copy to
# @param[in]   src Data being copy from
# @param[in]   sizeBytes Data size in bytes
# @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyDtoD(hipDeviceptr_t dst,hipDeviceptr_t src,int sizeBytes) nogil:
    global _lib_handle
    global hipMemcpyDtoD_funptr
    if hipMemcpyDtoD_funptr == NULL:
        with gil:
            hipMemcpyDtoD_funptr = loader.load_symbol(_lib_handle, "hipMemcpyDtoD")
    return (<hipError_t (*)(hipDeviceptr_t,hipDeviceptr_t,int) nogil> hipMemcpyDtoD_funptr)(dst,src,sizeBytes)


cdef void* hipMemcpyHtoDAsync_funptr = NULL
# @brief Copy data from Host to Device asynchronously
# @param[out]  dst Data being copy to
# @param[in]   src Data being copy from
# @param[in]   sizeBytes Data size in bytes
# @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst,void * src,int sizeBytes,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyHtoDAsync_funptr
    if hipMemcpyHtoDAsync_funptr == NULL:
        with gil:
            hipMemcpyHtoDAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpyHtoDAsync")
    return (<hipError_t (*)(hipDeviceptr_t,void *,int,hipStream_t) nogil> hipMemcpyHtoDAsync_funptr)(dst,src,sizeBytes,stream)


cdef void* hipMemcpyDtoHAsync_funptr = NULL
# @brief Copy data from Device to Host asynchronously
# @param[out]  dst Data being copy to
# @param[in]   src Data being copy from
# @param[in]   sizeBytes Data size in bytes
# @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyDtoHAsync(void * dst,hipDeviceptr_t src,int sizeBytes,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyDtoHAsync_funptr
    if hipMemcpyDtoHAsync_funptr == NULL:
        with gil:
            hipMemcpyDtoHAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpyDtoHAsync")
    return (<hipError_t (*)(void *,hipDeviceptr_t,int,hipStream_t) nogil> hipMemcpyDtoHAsync_funptr)(dst,src,sizeBytes,stream)


cdef void* hipMemcpyDtoDAsync_funptr = NULL
# @brief Copy data from Device to Device asynchronously
# @param[out]  dst Data being copy to
# @param[in]   src Data being copy from
# @param[in]   sizeBytes Data size in bytes
# @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst,hipDeviceptr_t src,int sizeBytes,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyDtoDAsync_funptr
    if hipMemcpyDtoDAsync_funptr == NULL:
        with gil:
            hipMemcpyDtoDAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpyDtoDAsync")
    return (<hipError_t (*)(hipDeviceptr_t,hipDeviceptr_t,int,hipStream_t) nogil> hipMemcpyDtoDAsync_funptr)(dst,src,sizeBytes,stream)


cdef void* hipModuleGetGlobal_funptr = NULL
# @brief Returns a global pointer from a module.
# Returns in *dptr and *bytes the pointer and size of the global of name name located in module hmod.
# If no variable of that name exists, it returns hipErrorNotFound. Both parameters dptr and bytes are optional.
# If one of them is NULL, it is ignored and hipSuccess is returned.
# @param[out]  dptr  Returns global device pointer
# @param[out]  bytes Returns global size in bytes
# @param[in]   hmod  Module to retrieve global from
# @param[in]   name  Name of global to retrieve
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotFound, #hipErrorInvalidContext
cdef hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr,int * bytes,hipModule_t hmod,const char * name) nogil:
    global _lib_handle
    global hipModuleGetGlobal_funptr
    if hipModuleGetGlobal_funptr == NULL:
        with gil:
            hipModuleGetGlobal_funptr = loader.load_symbol(_lib_handle, "hipModuleGetGlobal")
    return (<hipError_t (*)(hipDeviceptr_t*,int *,hipModule_t,const char *) nogil> hipModuleGetGlobal_funptr)(dptr,bytes,hmod,name)


cdef void* hipGetSymbolAddress_funptr = NULL
# @brief Gets device pointer associated with symbol on the device.
# @param[out]  devPtr  pointer to the device associated the symbole
# @param[in]   symbol  pointer to the symbole of the device
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipGetSymbolAddress(void ** devPtr,const void * symbol) nogil:
    global _lib_handle
    global hipGetSymbolAddress_funptr
    if hipGetSymbolAddress_funptr == NULL:
        with gil:
            hipGetSymbolAddress_funptr = loader.load_symbol(_lib_handle, "hipGetSymbolAddress")
    return (<hipError_t (*)(void **,const void *) nogil> hipGetSymbolAddress_funptr)(devPtr,symbol)


cdef void* hipGetSymbolSize_funptr = NULL
# @brief Gets the size of the given symbol on the device.
# @param[in]   symbol  pointer to the device symbole
# @param[out]  size  pointer to the size
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipGetSymbolSize(int * size,const void * symbol) nogil:
    global _lib_handle
    global hipGetSymbolSize_funptr
    if hipGetSymbolSize_funptr == NULL:
        with gil:
            hipGetSymbolSize_funptr = loader.load_symbol(_lib_handle, "hipGetSymbolSize")
    return (<hipError_t (*)(int *,const void *) nogil> hipGetSymbolSize_funptr)(size,symbol)


cdef void* hipMemcpyToSymbol_funptr = NULL
# @brief Copies data to the given symbol on the device.
# Symbol HIP APIs allow a kernel to define a device-side data symbol which can be accessed on
# the host side. The symbol can be in __constant or device space.
# Note that the symbol name needs to be encased in the HIP_SYMBOL macro.
# This also applies to hipMemcpyFromSymbol, hipGetSymbolAddress, and hipGetSymbolSize.
# For detail usage, see the example at
# https://github.com/ROCm-Developer-Tools/HIP/blob/rocm-5.0.x/docs/markdown/hip_porting_guide.md
# @param[out]  symbol  pointer to the device symbole
# @param[in]   src  pointer to the source address
# @param[in]   sizeBytes  size in bytes to copy
# @param[in]   offset  offset in bytes from start of symbole
# @param[in]   kind  type of memory transfer
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemcpyToSymbol(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpyToSymbol_funptr
    if hipMemcpyToSymbol_funptr == NULL:
        with gil:
            hipMemcpyToSymbol_funptr = loader.load_symbol(_lib_handle, "hipMemcpyToSymbol")
    return (<hipError_t (*)(const void *,const void *,int,int,hipMemcpyKind) nogil> hipMemcpyToSymbol_funptr)(symbol,src,sizeBytes,offset,kind)


cdef void* hipMemcpyToSymbolAsync_funptr = NULL
# @brief Copies data to the given symbol on the device asynchronously.
# @param[out]  symbol  pointer to the device symbole
# @param[in]   src  pointer to the source address
# @param[in]   sizeBytes  size in bytes to copy
# @param[in]   offset  offset in bytes from start of symbole
# @param[in]   kind  type of memory transfer
# @param[in]   stream  stream identifier
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemcpyToSymbolAsync(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyToSymbolAsync_funptr
    if hipMemcpyToSymbolAsync_funptr == NULL:
        with gil:
            hipMemcpyToSymbolAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpyToSymbolAsync")
    return (<hipError_t (*)(const void *,const void *,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpyToSymbolAsync_funptr)(symbol,src,sizeBytes,offset,kind,stream)


cdef void* hipMemcpyFromSymbol_funptr = NULL
# @brief Copies data from the given symbol on the device.
# @param[out]  dptr  Returns pointer to destinition memory address
# @param[in]   symbol  pointer to the symbole address on the device
# @param[in]   sizeBytes  size in bytes to copy
# @param[in]   offset  offset in bytes from the start of symbole
# @param[in]   kind  type of memory transfer
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemcpyFromSymbol(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpyFromSymbol_funptr
    if hipMemcpyFromSymbol_funptr == NULL:
        with gil:
            hipMemcpyFromSymbol_funptr = loader.load_symbol(_lib_handle, "hipMemcpyFromSymbol")
    return (<hipError_t (*)(void *,const void *,int,int,hipMemcpyKind) nogil> hipMemcpyFromSymbol_funptr)(dst,symbol,sizeBytes,offset,kind)


cdef void* hipMemcpyFromSymbolAsync_funptr = NULL
# @brief Copies data from the given symbol on the device asynchronously.
# @param[out]  dptr  Returns pointer to destinition memory address
# @param[in]   symbol  pointer to the symbole address on the device
# @param[in]   sizeBytes  size in bytes to copy
# @param[in]   offset  offset in bytes from the start of symbole
# @param[in]   kind  type of memory transfer
# @param[in]   stream  stream identifier
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemcpyFromSymbolAsync(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyFromSymbolAsync_funptr
    if hipMemcpyFromSymbolAsync_funptr == NULL:
        with gil:
            hipMemcpyFromSymbolAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpyFromSymbolAsync")
    return (<hipError_t (*)(void *,const void *,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpyFromSymbolAsync_funptr)(dst,symbol,sizeBytes,offset,kind,stream)


cdef void* hipMemcpyAsync_funptr = NULL
# @brief Copy data from src to dst asynchronously.
# @warning If host or dest are not pinned, the memory copy will be performed synchronously.  For
# best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.
# @warning on HCC hipMemcpyAsync does not support overlapped H2D and D2H copies.
# For hipMemcpy, the copy is always performed by the device associated with the specified stream.
# For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is a
# attached to the device where the src data is physically located. For optimal peer-to-peer copies,
# the copy device must be able to access the src and dst pointers (by calling
# hipDeviceEnablePeerAccess with copy agent as the current device and src/dest as the peerDevice
# argument.  if this is not done, the hipMemcpy will still work, but will perform the copy using a
# staging buffer on the host.
# @param[out] dst Data being copy to
# @param[in]  src Data being copy from
# @param[in]  sizeBytes Data size in bytes
# @param[in]  accelerator_view Accelerator view which the copy is being enqueued
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
# @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
# hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyToSymbol,
# hipMemcpyFromSymbol, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync,
# hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync,
# hipMemcpyFromSymbolAsync
cdef hipError_t hipMemcpyAsync(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyAsync_funptr
    if hipMemcpyAsync_funptr == NULL:
        with gil:
            hipMemcpyAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpyAsync")
    return (<hipError_t (*)(void *,const void *,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpyAsync_funptr)(dst,src,sizeBytes,kind,stream)


cdef void* hipMemset_funptr = NULL
# @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# byte value value.
# @param[out] dst Data being filled
# @param[in]  constant value to be set
# @param[in]  sizeBytes Data size in bytes
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemset(void * dst,int value,int sizeBytes) nogil:
    global _lib_handle
    global hipMemset_funptr
    if hipMemset_funptr == NULL:
        with gil:
            hipMemset_funptr = loader.load_symbol(_lib_handle, "hipMemset")
    return (<hipError_t (*)(void *,int,int) nogil> hipMemset_funptr)(dst,value,sizeBytes)


cdef void* hipMemsetD8_funptr = NULL
# @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# byte value value.
# @param[out] dst Data ptr to be filled
# @param[in]  constant value to be set
# @param[in]  number of values to be set
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD8(hipDeviceptr_t dest,unsigned char value,int count) nogil:
    global _lib_handle
    global hipMemsetD8_funptr
    if hipMemsetD8_funptr == NULL:
        with gil:
            hipMemsetD8_funptr = loader.load_symbol(_lib_handle, "hipMemsetD8")
    return (<hipError_t (*)(hipDeviceptr_t,unsigned char,int) nogil> hipMemsetD8_funptr)(dest,value,count)


cdef void* hipMemsetD8Async_funptr = NULL
# @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# byte value value.
# hipMemsetD8Async() is asynchronous with respect to the host, so the call may return before the
# memset is complete. The operation can optionally be associated to a stream by passing a non-zero
# stream argument. If stream is non-zero, the operation may overlap with operations in other
# streams.
# @param[out] dst Data ptr to be filled
# @param[in]  constant value to be set
# @param[in]  number of values to be set
# @param[in]  stream - Stream identifier
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD8Async(hipDeviceptr_t dest,unsigned char value,int count,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemsetD8Async_funptr
    if hipMemsetD8Async_funptr == NULL:
        with gil:
            hipMemsetD8Async_funptr = loader.load_symbol(_lib_handle, "hipMemsetD8Async")
    return (<hipError_t (*)(hipDeviceptr_t,unsigned char,int,hipStream_t) nogil> hipMemsetD8Async_funptr)(dest,value,count,stream)


cdef void* hipMemsetD16_funptr = NULL
# @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# short value value.
# @param[out] dst Data ptr to be filled
# @param[in]  constant value to be set
# @param[in]  number of values to be set
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD16(hipDeviceptr_t dest,unsigned short value,int count) nogil:
    global _lib_handle
    global hipMemsetD16_funptr
    if hipMemsetD16_funptr == NULL:
        with gil:
            hipMemsetD16_funptr = loader.load_symbol(_lib_handle, "hipMemsetD16")
    return (<hipError_t (*)(hipDeviceptr_t,unsigned short,int) nogil> hipMemsetD16_funptr)(dest,value,count)


cdef void* hipMemsetD16Async_funptr = NULL
# @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# short value value.
# hipMemsetD16Async() is asynchronous with respect to the host, so the call may return before the
# memset is complete. The operation can optionally be associated to a stream by passing a non-zero
# stream argument. If stream is non-zero, the operation may overlap with operations in other
# streams.
# @param[out] dst Data ptr to be filled
# @param[in]  constant value to be set
# @param[in]  number of values to be set
# @param[in]  stream - Stream identifier
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD16Async(hipDeviceptr_t dest,unsigned short value,int count,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemsetD16Async_funptr
    if hipMemsetD16Async_funptr == NULL:
        with gil:
            hipMemsetD16Async_funptr = loader.load_symbol(_lib_handle, "hipMemsetD16Async")
    return (<hipError_t (*)(hipDeviceptr_t,unsigned short,int,hipStream_t) nogil> hipMemsetD16Async_funptr)(dest,value,count,stream)


cdef void* hipMemsetD32_funptr = NULL
# @brief Fills the memory area pointed to by dest with the constant integer
# value for specified number of times.
# @param[out] dst Data being filled
# @param[in]  constant value to be set
# @param[in]  number of values to be set
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD32(hipDeviceptr_t dest,int value,int count) nogil:
    global _lib_handle
    global hipMemsetD32_funptr
    if hipMemsetD32_funptr == NULL:
        with gil:
            hipMemsetD32_funptr = loader.load_symbol(_lib_handle, "hipMemsetD32")
    return (<hipError_t (*)(hipDeviceptr_t,int,int) nogil> hipMemsetD32_funptr)(dest,value,count)


cdef void* hipMemsetAsync_funptr = NULL
# @brief Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant
# byte value value.
# hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the
# memset is complete. The operation can optionally be associated to a stream by passing a non-zero
# stream argument. If stream is non-zero, the operation may overlap with operations in other
# streams.
# @param[out] dst Pointer to device memory
# @param[in]  value - Value to set for each byte of specified memory
# @param[in]  sizeBytes - Size in bytes to set
# @param[in]  stream - Stream identifier
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemsetAsync(void * dst,int value,int sizeBytes,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemsetAsync_funptr
    if hipMemsetAsync_funptr == NULL:
        with gil:
            hipMemsetAsync_funptr = loader.load_symbol(_lib_handle, "hipMemsetAsync")
    return (<hipError_t (*)(void *,int,int,hipStream_t) nogil> hipMemsetAsync_funptr)(dst,value,sizeBytes,stream)


cdef void* hipMemsetD32Async_funptr = NULL
# @brief Fills the memory area pointed to by dev with the constant integer
# value for specified number of times.
# hipMemsetD32Async() is asynchronous with respect to the host, so the call may return before the
# memset is complete. The operation can optionally be associated to a stream by passing a non-zero
# stream argument. If stream is non-zero, the operation may overlap with operations in other
# streams.
# @param[out] dst Pointer to device memory
# @param[in]  value - Value to set for each byte of specified memory
# @param[in]  count - number of values to be set
# @param[in]  stream - Stream identifier
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemsetD32Async(hipDeviceptr_t dst,int value,int count,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemsetD32Async_funptr
    if hipMemsetD32Async_funptr == NULL:
        with gil:
            hipMemsetD32Async_funptr = loader.load_symbol(_lib_handle, "hipMemsetD32Async")
    return (<hipError_t (*)(hipDeviceptr_t,int,int,hipStream_t) nogil> hipMemsetD32Async_funptr)(dst,value,count,stream)


cdef void* hipMemset2D_funptr = NULL
# @brief Fills the memory area pointed to by dst with the constant value.
# @param[out] dst Pointer to device memory
# @param[in]  pitch - data size in bytes
# @param[in]  value - constant value to be set
# @param[in]  width
# @param[in]  height
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemset2D(void * dst,int pitch,int value,int width,int height) nogil:
    global _lib_handle
    global hipMemset2D_funptr
    if hipMemset2D_funptr == NULL:
        with gil:
            hipMemset2D_funptr = loader.load_symbol(_lib_handle, "hipMemset2D")
    return (<hipError_t (*)(void *,int,int,int,int) nogil> hipMemset2D_funptr)(dst,pitch,value,width,height)


cdef void* hipMemset2DAsync_funptr = NULL
# @brief Fills asynchronously the memory area pointed to by dst with the constant value.
# @param[in]  dst Pointer to device memory
# @param[in]  pitch - data size in bytes
# @param[in]  value - constant value to be set
# @param[in]  width
# @param[in]  height
# @param[in]  stream
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemset2DAsync(void * dst,int pitch,int value,int width,int height,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemset2DAsync_funptr
    if hipMemset2DAsync_funptr == NULL:
        with gil:
            hipMemset2DAsync_funptr = loader.load_symbol(_lib_handle, "hipMemset2DAsync")
    return (<hipError_t (*)(void *,int,int,int,int,hipStream_t) nogil> hipMemset2DAsync_funptr)(dst,pitch,value,width,height,stream)


cdef void* hipMemset3D_funptr = NULL
# @brief Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
# @param[in] pitchedDevPtr
# @param[in]  value - constant value to be set
# @param[in]  extent
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent) nogil:
    global _lib_handle
    global hipMemset3D_funptr
    if hipMemset3D_funptr == NULL:
        with gil:
            hipMemset3D_funptr = loader.load_symbol(_lib_handle, "hipMemset3D")
    return (<hipError_t (*)(hipPitchedPtr,int,hipExtent) nogil> hipMemset3D_funptr)(pitchedDevPtr,value,extent)


cdef void* hipMemset3DAsync_funptr = NULL
# @brief Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
# @param[in] pitchedDevPtr
# @param[in]  value - constant value to be set
# @param[in]  extent
# @param[in]  stream
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemset3DAsync_funptr
    if hipMemset3DAsync_funptr == NULL:
        with gil:
            hipMemset3DAsync_funptr = loader.load_symbol(_lib_handle, "hipMemset3DAsync")
    return (<hipError_t (*)(hipPitchedPtr,int,hipExtent,hipStream_t) nogil> hipMemset3DAsync_funptr)(pitchedDevPtr,value,extent,stream)


cdef void* hipMemGetInfo_funptr = NULL
# @brief Query memory info.
# Return snapshot of free memory, and total allocatable memory on the device.
# Returns in *free a snapshot of the current free memory.
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# @warning On HCC, the free memory only accounts for memory allocated by this process and may be
# optimistic.
cdef hipError_t hipMemGetInfo(int * free,int * total) nogil:
    global _lib_handle
    global hipMemGetInfo_funptr
    if hipMemGetInfo_funptr == NULL:
        with gil:
            hipMemGetInfo_funptr = loader.load_symbol(_lib_handle, "hipMemGetInfo")
    return (<hipError_t (*)(int *,int *) nogil> hipMemGetInfo_funptr)(free,total)


cdef void* hipMemPtrGetInfo_funptr = NULL
cdef hipError_t hipMemPtrGetInfo(void * ptr,int * size) nogil:
    global _lib_handle
    global hipMemPtrGetInfo_funptr
    if hipMemPtrGetInfo_funptr == NULL:
        with gil:
            hipMemPtrGetInfo_funptr = loader.load_symbol(_lib_handle, "hipMemPtrGetInfo")
    return (<hipError_t (*)(void *,int *) nogil> hipMemPtrGetInfo_funptr)(ptr,size)


cdef void* hipMallocArray_funptr = NULL
# @brief Allocate an array on the device.
# @param[out]  array  Pointer to allocated array in device memory
# @param[in]   desc   Requested channel format
# @param[in]   width  Requested array allocation width
# @param[in]   height Requested array allocation height
# @param[in]   flags  Requested properties of allocated array
# @return      #hipSuccess, #hipErrorOutOfMemory
# @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
cdef hipError_t hipMallocArray(hipArray ** array,hipChannelFormatDesc * desc,int width,int height,unsigned int flags) nogil:
    global _lib_handle
    global hipMallocArray_funptr
    if hipMallocArray_funptr == NULL:
        with gil:
            hipMallocArray_funptr = loader.load_symbol(_lib_handle, "hipMallocArray")
    return (<hipError_t (*)(hipArray **,hipChannelFormatDesc *,int,int,unsigned int) nogil> hipMallocArray_funptr)(array,desc,width,height,flags)


cdef void* hipArrayCreate_funptr = NULL
cdef hipError_t hipArrayCreate(hipArray ** pHandle,HIP_ARRAY_DESCRIPTOR * pAllocateArray) nogil:
    global _lib_handle
    global hipArrayCreate_funptr
    if hipArrayCreate_funptr == NULL:
        with gil:
            hipArrayCreate_funptr = loader.load_symbol(_lib_handle, "hipArrayCreate")
    return (<hipError_t (*)(hipArray **,HIP_ARRAY_DESCRIPTOR *) nogil> hipArrayCreate_funptr)(pHandle,pAllocateArray)


cdef void* hipArrayDestroy_funptr = NULL
cdef hipError_t hipArrayDestroy(hipArray * array) nogil:
    global _lib_handle
    global hipArrayDestroy_funptr
    if hipArrayDestroy_funptr == NULL:
        with gil:
            hipArrayDestroy_funptr = loader.load_symbol(_lib_handle, "hipArrayDestroy")
    return (<hipError_t (*)(hipArray *) nogil> hipArrayDestroy_funptr)(array)


cdef void* hipArray3DCreate_funptr = NULL
cdef hipError_t hipArray3DCreate(hipArray ** array,HIP_ARRAY3D_DESCRIPTOR * pAllocateArray) nogil:
    global _lib_handle
    global hipArray3DCreate_funptr
    if hipArray3DCreate_funptr == NULL:
        with gil:
            hipArray3DCreate_funptr = loader.load_symbol(_lib_handle, "hipArray3DCreate")
    return (<hipError_t (*)(hipArray **,HIP_ARRAY3D_DESCRIPTOR *) nogil> hipArray3DCreate_funptr)(array,pAllocateArray)


cdef void* hipMalloc3D_funptr = NULL
cdef hipError_t hipMalloc3D(hipPitchedPtr * pitchedDevPtr,hipExtent extent) nogil:
    global _lib_handle
    global hipMalloc3D_funptr
    if hipMalloc3D_funptr == NULL:
        with gil:
            hipMalloc3D_funptr = loader.load_symbol(_lib_handle, "hipMalloc3D")
    return (<hipError_t (*)(hipPitchedPtr *,hipExtent) nogil> hipMalloc3D_funptr)(pitchedDevPtr,extent)


cdef void* hipFreeArray_funptr = NULL
# @brief Frees an array on the device.
# @param[in]  array  Pointer to array to free
# @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
# @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
cdef hipError_t hipFreeArray(hipArray * array) nogil:
    global _lib_handle
    global hipFreeArray_funptr
    if hipFreeArray_funptr == NULL:
        with gil:
            hipFreeArray_funptr = loader.load_symbol(_lib_handle, "hipFreeArray")
    return (<hipError_t (*)(hipArray *) nogil> hipFreeArray_funptr)(array)


cdef void* hipFreeMipmappedArray_funptr = NULL
# @brief Frees a mipmapped array on the device
# @param[in] mipmappedArray - Pointer to mipmapped array to free
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) nogil:
    global _lib_handle
    global hipFreeMipmappedArray_funptr
    if hipFreeMipmappedArray_funptr == NULL:
        with gil:
            hipFreeMipmappedArray_funptr = loader.load_symbol(_lib_handle, "hipFreeMipmappedArray")
    return (<hipError_t (*)(hipMipmappedArray_t) nogil> hipFreeMipmappedArray_funptr)(mipmappedArray)


cdef void* hipMalloc3DArray_funptr = NULL
# @brief Allocate an array on the device.
# @param[out]  array  Pointer to allocated array in device memory
# @param[in]   desc   Requested channel format
# @param[in]   extent Requested array allocation width, height and depth
# @param[in]   flags  Requested properties of allocated array
# @return      #hipSuccess, #hipErrorOutOfMemory
# @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
cdef hipError_t hipMalloc3DArray(hipArray ** array,hipChannelFormatDesc * desc,hipExtent extent,unsigned int flags) nogil:
    global _lib_handle
    global hipMalloc3DArray_funptr
    if hipMalloc3DArray_funptr == NULL:
        with gil:
            hipMalloc3DArray_funptr = loader.load_symbol(_lib_handle, "hipMalloc3DArray")
    return (<hipError_t (*)(hipArray **,hipChannelFormatDesc *,hipExtent,unsigned int) nogil> hipMalloc3DArray_funptr)(array,desc,extent,flags)


cdef void* hipMallocMipmappedArray_funptr = NULL
# @brief Allocate a mipmapped array on the device
# @param[out] mipmappedArray  - Pointer to allocated mipmapped array in device memory
# @param[in]  desc            - Requested channel format
# @param[in]  extent          - Requested allocation size (width field in elements)
# @param[in]  numLevels       - Number of mipmap levels to allocate
# @param[in]  flags           - Flags for extensions
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
cdef hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray,hipChannelFormatDesc * desc,hipExtent extent,unsigned int numLevels,unsigned int flags) nogil:
    global _lib_handle
    global hipMallocMipmappedArray_funptr
    if hipMallocMipmappedArray_funptr == NULL:
        with gil:
            hipMallocMipmappedArray_funptr = loader.load_symbol(_lib_handle, "hipMallocMipmappedArray")
    return (<hipError_t (*)(hipMipmappedArray_t*,hipChannelFormatDesc *,hipExtent,unsigned int,unsigned int) nogil> hipMallocMipmappedArray_funptr)(mipmappedArray,desc,extent,numLevels,flags)


cdef void* hipGetMipmappedArrayLevel_funptr = NULL
# @brief Gets a mipmap level of a HIP mipmapped array
# @param[out] levelArray     - Returned mipmap level HIP array
# @param[in]  mipmappedArray - HIP mipmapped array
# @param[in]  level          - Mipmap level
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray,hipMipmappedArray_const_t mipmappedArray,unsigned int level) nogil:
    global _lib_handle
    global hipGetMipmappedArrayLevel_funptr
    if hipGetMipmappedArrayLevel_funptr == NULL:
        with gil:
            hipGetMipmappedArrayLevel_funptr = loader.load_symbol(_lib_handle, "hipGetMipmappedArrayLevel")
    return (<hipError_t (*)(hipArray_t*,hipMipmappedArray_const_t,unsigned int) nogil> hipGetMipmappedArrayLevel_funptr)(levelArray,mipmappedArray,level)


cdef void* hipMemcpy2D_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   dst    Destination memory address
# @param[in]   dpitch Pitch of destination memory
# @param[in]   src    Source memory address
# @param[in]   spitch Pitch of source memory
# @param[in]   width  Width of matrix transfer (columns in bytes)
# @param[in]   height Height of matrix transfer (rows)
# @param[in]   kind   Type of transfer
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2D(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpy2D_funptr
    if hipMemcpy2D_funptr == NULL:
        with gil:
            hipMemcpy2D_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2D")
    return (<hipError_t (*)(void *,int,const void *,int,int,int,hipMemcpyKind) nogil> hipMemcpy2D_funptr)(dst,dpitch,src,spitch,width,height,kind)


cdef void* hipMemcpyParam2D_funptr = NULL
# @brief Copies memory for 2D arrays.
# @param[in]   pCopy Parameters for the memory copy
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
# hipMemcpyToSymbol, hipMemcpyAsync
cdef hipError_t hipMemcpyParam2D(hip_Memcpy2D * pCopy) nogil:
    global _lib_handle
    global hipMemcpyParam2D_funptr
    if hipMemcpyParam2D_funptr == NULL:
        with gil:
            hipMemcpyParam2D_funptr = loader.load_symbol(_lib_handle, "hipMemcpyParam2D")
    return (<hipError_t (*)(hip_Memcpy2D *) nogil> hipMemcpyParam2D_funptr)(pCopy)


cdef void* hipMemcpyParam2DAsync_funptr = NULL
# @brief Copies memory for 2D arrays.
# @param[in]   pCopy Parameters for the memory copy
# @param[in]   stream Stream to use
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
# hipMemcpyToSymbol, hipMemcpyAsync
cdef hipError_t hipMemcpyParam2DAsync(hip_Memcpy2D * pCopy,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyParam2DAsync_funptr
    if hipMemcpyParam2DAsync_funptr == NULL:
        with gil:
            hipMemcpyParam2DAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpyParam2DAsync")
    return (<hipError_t (*)(hip_Memcpy2D *,hipStream_t) nogil> hipMemcpyParam2DAsync_funptr)(pCopy,stream)


cdef void* hipMemcpy2DAsync_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   dst    Destination memory address
# @param[in]   dpitch Pitch of destination memory
# @param[in]   src    Source memory address
# @param[in]   spitch Pitch of source memory
# @param[in]   width  Width of matrix transfer (columns in bytes)
# @param[in]   height Height of matrix transfer (rows)
# @param[in]   kind   Type of transfer
# @param[in]   stream Stream to use
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DAsync(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpy2DAsync_funptr
    if hipMemcpy2DAsync_funptr == NULL:
        with gil:
            hipMemcpy2DAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DAsync")
    return (<hipError_t (*)(void *,int,const void *,int,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpy2DAsync_funptr)(dst,dpitch,src,spitch,width,height,kind,stream)


cdef void* hipMemcpy2DToArray_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   dst     Destination memory address
# @param[in]   wOffset Destination starting X offset
# @param[in]   hOffset Destination starting Y offset
# @param[in]   src     Source memory address
# @param[in]   spitch  Pitch of source memory
# @param[in]   width   Width of matrix transfer (columns in bytes)
# @param[in]   height  Height of matrix transfer (rows)
# @param[in]   kind    Type of transfer
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DToArray(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpy2DToArray_funptr
    if hipMemcpy2DToArray_funptr == NULL:
        with gil:
            hipMemcpy2DToArray_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DToArray")
    return (<hipError_t (*)(hipArray *,int,int,const void *,int,int,int,hipMemcpyKind) nogil> hipMemcpy2DToArray_funptr)(dst,wOffset,hOffset,src,spitch,width,height,kind)


cdef void* hipMemcpy2DToArrayAsync_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   dst     Destination memory address
# @param[in]   wOffset Destination starting X offset
# @param[in]   hOffset Destination starting Y offset
# @param[in]   src     Source memory address
# @param[in]   spitch  Pitch of source memory
# @param[in]   width   Width of matrix transfer (columns in bytes)
# @param[in]   height  Height of matrix transfer (rows)
# @param[in]   kind    Type of transfer
# @param[in]   stream    Accelerator view which the copy is being enqueued
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DToArrayAsync(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpy2DToArrayAsync_funptr
    if hipMemcpy2DToArrayAsync_funptr == NULL:
        with gil:
            hipMemcpy2DToArrayAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DToArrayAsync")
    return (<hipError_t (*)(hipArray *,int,int,const void *,int,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpy2DToArrayAsync_funptr)(dst,wOffset,hOffset,src,spitch,width,height,kind,stream)


cdef void* hipMemcpyToArray_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   dst     Destination memory address
# @param[in]   wOffset Destination starting X offset
# @param[in]   hOffset Destination starting Y offset
# @param[in]   src     Source memory address
# @param[in]   count   size in bytes to copy
# @param[in]   kind    Type of transfer
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpyToArray(hipArray * dst,int wOffset,int hOffset,const void * src,int count,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpyToArray_funptr
    if hipMemcpyToArray_funptr == NULL:
        with gil:
            hipMemcpyToArray_funptr = loader.load_symbol(_lib_handle, "hipMemcpyToArray")
    return (<hipError_t (*)(hipArray *,int,int,const void *,int,hipMemcpyKind) nogil> hipMemcpyToArray_funptr)(dst,wOffset,hOffset,src,count,kind)


cdef void* hipMemcpyFromArray_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   dst       Destination memory address
# @param[in]   srcArray  Source memory address
# @param[in]   woffset   Source starting X offset
# @param[in]   hOffset   Source starting Y offset
# @param[in]   count     Size in bytes to copy
# @param[in]   kind      Type of transfer
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpyFromArray(void * dst,hipArray_const_t srcArray,int wOffset,int hOffset,int count,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpyFromArray_funptr
    if hipMemcpyFromArray_funptr == NULL:
        with gil:
            hipMemcpyFromArray_funptr = loader.load_symbol(_lib_handle, "hipMemcpyFromArray")
    return (<hipError_t (*)(void *,hipArray_const_t,int,int,int,hipMemcpyKind) nogil> hipMemcpyFromArray_funptr)(dst,srcArray,wOffset,hOffset,count,kind)


cdef void* hipMemcpy2DFromArray_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   dst       Destination memory address
# @param[in]   dpitch    Pitch of destination memory
# @param[in]   src       Source memory address
# @param[in]   wOffset   Source starting X offset
# @param[in]   hOffset   Source starting Y offset
# @param[in]   width     Width of matrix transfer (columns in bytes)
# @param[in]   height    Height of matrix transfer (rows)
# @param[in]   kind      Type of transfer
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DFromArray(void * dst,int dpitch,hipArray_const_t src,int wOffset,int hOffset,int width,int height,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpy2DFromArray_funptr
    if hipMemcpy2DFromArray_funptr == NULL:
        with gil:
            hipMemcpy2DFromArray_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DFromArray")
    return (<hipError_t (*)(void *,int,hipArray_const_t,int,int,int,int,hipMemcpyKind) nogil> hipMemcpy2DFromArray_funptr)(dst,dpitch,src,wOffset,hOffset,width,height,kind)


cdef void* hipMemcpy2DFromArrayAsync_funptr = NULL
# @brief Copies data between host and device asynchronously.
# @param[in]   dst       Destination memory address
# @param[in]   dpitch    Pitch of destination memory
# @param[in]   src       Source memory address
# @param[in]   wOffset   Source starting X offset
# @param[in]   hOffset   Source starting Y offset
# @param[in]   width     Width of matrix transfer (columns in bytes)
# @param[in]   height    Height of matrix transfer (rows)
# @param[in]   kind      Type of transfer
# @param[in]   stream    Accelerator view which the copy is being enqueued
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DFromArrayAsync(void * dst,int dpitch,hipArray_const_t src,int wOffset,int hOffset,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpy2DFromArrayAsync_funptr
    if hipMemcpy2DFromArrayAsync_funptr == NULL:
        with gil:
            hipMemcpy2DFromArrayAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DFromArrayAsync")
    return (<hipError_t (*)(void *,int,hipArray_const_t,int,int,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpy2DFromArrayAsync_funptr)(dst,dpitch,src,wOffset,hOffset,width,height,kind,stream)


cdef void* hipMemcpyAtoH_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   dst       Destination memory address
# @param[in]   srcArray  Source array
# @param[in]   srcoffset Offset in bytes of source array
# @param[in]   count     Size of memory copy in bytes
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpyAtoH(void * dst,hipArray * srcArray,int srcOffset,int count) nogil:
    global _lib_handle
    global hipMemcpyAtoH_funptr
    if hipMemcpyAtoH_funptr == NULL:
        with gil:
            hipMemcpyAtoH_funptr = loader.load_symbol(_lib_handle, "hipMemcpyAtoH")
    return (<hipError_t (*)(void *,hipArray *,int,int) nogil> hipMemcpyAtoH_funptr)(dst,srcArray,srcOffset,count)


cdef void* hipMemcpyHtoA_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   dstArray   Destination memory address
# @param[in]   dstOffset  Offset in bytes of destination array
# @param[in]   srcHost    Source host pointer
# @param[in]   count      Size of memory copy in bytes
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpyHtoA(hipArray * dstArray,int dstOffset,const void * srcHost,int count) nogil:
    global _lib_handle
    global hipMemcpyHtoA_funptr
    if hipMemcpyHtoA_funptr == NULL:
        with gil:
            hipMemcpyHtoA_funptr = loader.load_symbol(_lib_handle, "hipMemcpyHtoA")
    return (<hipError_t (*)(hipArray *,int,const void *,int) nogil> hipMemcpyHtoA_funptr)(dstArray,dstOffset,srcHost,count)


cdef void* hipMemcpy3D_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   p   3D memory copy parameters
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy3D(hipMemcpy3DParms * p) nogil:
    global _lib_handle
    global hipMemcpy3D_funptr
    if hipMemcpy3D_funptr == NULL:
        with gil:
            hipMemcpy3D_funptr = loader.load_symbol(_lib_handle, "hipMemcpy3D")
    return (<hipError_t (*)(hipMemcpy3DParms *) nogil> hipMemcpy3D_funptr)(p)


cdef void* hipMemcpy3DAsync_funptr = NULL
# @brief Copies data between host and device asynchronously.
# @param[in]   p        3D memory copy parameters
# @param[in]   stream   Stream to use
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy3DAsync(hipMemcpy3DParms * p,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpy3DAsync_funptr
    if hipMemcpy3DAsync_funptr == NULL:
        with gil:
            hipMemcpy3DAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpy3DAsync")
    return (<hipError_t (*)(hipMemcpy3DParms *,hipStream_t) nogil> hipMemcpy3DAsync_funptr)(p,stream)


cdef void* hipDrvMemcpy3D_funptr = NULL
# @brief Copies data between host and device.
# @param[in]   pCopy   3D memory copy parameters
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipDrvMemcpy3D(HIP_MEMCPY3D * pCopy) nogil:
    global _lib_handle
    global hipDrvMemcpy3D_funptr
    if hipDrvMemcpy3D_funptr == NULL:
        with gil:
            hipDrvMemcpy3D_funptr = loader.load_symbol(_lib_handle, "hipDrvMemcpy3D")
    return (<hipError_t (*)(HIP_MEMCPY3D *) nogil> hipDrvMemcpy3D_funptr)(pCopy)


cdef void* hipDrvMemcpy3DAsync_funptr = NULL
# @brief Copies data between host and device asynchronously.
# @param[in]   pCopy    3D memory copy parameters
# @param[in]   stream   Stream to use
# @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipDrvMemcpy3DAsync(HIP_MEMCPY3D * pCopy,hipStream_t stream) nogil:
    global _lib_handle
    global hipDrvMemcpy3DAsync_funptr
    if hipDrvMemcpy3DAsync_funptr == NULL:
        with gil:
            hipDrvMemcpy3DAsync_funptr = loader.load_symbol(_lib_handle, "hipDrvMemcpy3DAsync")
    return (<hipError_t (*)(HIP_MEMCPY3D *,hipStream_t) nogil> hipDrvMemcpy3DAsync_funptr)(pCopy,stream)


cdef void* hipDeviceCanAccessPeer_funptr = NULL
# @}
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# @defgroup PeerToPeer PeerToPeer Device Memory Access
# @{
# @warning PeerToPeer support is experimental.
# This section describes the PeerToPeer device memory access functions of HIP runtime API.
# @brief Determine if a device can access a peer's memory.
# @param [out] canAccessPeer Returns the peer access capability (0 or 1)
# @param [in] device - device from where memory may be accessed.
# @param [in] peerDevice - device where memory is physically located
# Returns "1" in @p canAccessPeer if the specified @p device is capable
# of directly accessing memory physically located on peerDevice , or "0" if not.
# Returns "0" in @p canAccessPeer if deviceId == peerDeviceId, and both are valid devices : a
# device is not a peer of itself.
# @returns #hipSuccess,
# @returns #hipErrorInvalidDevice if deviceId or peerDeviceId are not valid devices
cdef hipError_t hipDeviceCanAccessPeer(int * canAccessPeer,int deviceId,int peerDeviceId) nogil:
    global _lib_handle
    global hipDeviceCanAccessPeer_funptr
    if hipDeviceCanAccessPeer_funptr == NULL:
        with gil:
            hipDeviceCanAccessPeer_funptr = loader.load_symbol(_lib_handle, "hipDeviceCanAccessPeer")
    return (<hipError_t (*)(int *,int,int) nogil> hipDeviceCanAccessPeer_funptr)(canAccessPeer,deviceId,peerDeviceId)


cdef void* hipDeviceEnablePeerAccess_funptr = NULL
# @brief Enable direct access from current device's virtual address space to memory allocations
# physically located on a peer device.
# Memory which already allocated on peer device will be mapped into the address space of the
# current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
# the address space of the current device when the memory is allocated. The peer memory remains
# accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
# @param [in] peerDeviceId
# @param [in] flags
# Returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
# @returns #hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device.
cdef hipError_t hipDeviceEnablePeerAccess(int peerDeviceId,unsigned int flags) nogil:
    global _lib_handle
    global hipDeviceEnablePeerAccess_funptr
    if hipDeviceEnablePeerAccess_funptr == NULL:
        with gil:
            hipDeviceEnablePeerAccess_funptr = loader.load_symbol(_lib_handle, "hipDeviceEnablePeerAccess")
    return (<hipError_t (*)(int,unsigned int) nogil> hipDeviceEnablePeerAccess_funptr)(peerDeviceId,flags)


cdef void* hipDeviceDisablePeerAccess_funptr = NULL
# @brief Disable direct access from current device's virtual address space to memory allocations
# physically located on a peer device.
# Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
# enabled from the current device.
# @param [in] peerDeviceId
# @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
cdef hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) nogil:
    global _lib_handle
    global hipDeviceDisablePeerAccess_funptr
    if hipDeviceDisablePeerAccess_funptr == NULL:
        with gil:
            hipDeviceDisablePeerAccess_funptr = loader.load_symbol(_lib_handle, "hipDeviceDisablePeerAccess")
    return (<hipError_t (*)(int) nogil> hipDeviceDisablePeerAccess_funptr)(peerDeviceId)


cdef void* hipMemGetAddressRange_funptr = NULL
# @brief Get information on memory allocations.
# @param [out] pbase - BAse pointer address
# @param [out] psize - Size of allocation
# @param [in]  dptr- Device Pointer
# @returns #hipSuccess, #hipErrorInvalidDevicePointer
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase,int * psize,hipDeviceptr_t dptr) nogil:
    global _lib_handle
    global hipMemGetAddressRange_funptr
    if hipMemGetAddressRange_funptr == NULL:
        with gil:
            hipMemGetAddressRange_funptr = loader.load_symbol(_lib_handle, "hipMemGetAddressRange")
    return (<hipError_t (*)(hipDeviceptr_t*,int *,hipDeviceptr_t) nogil> hipMemGetAddressRange_funptr)(pbase,psize,dptr)


cdef void* hipMemcpyPeer_funptr = NULL
# @brief Copies memory from one device to memory on another device.
# @param [out] dst - Destination device pointer.
# @param [in] dstDeviceId - Destination device
# @param [in] src - Source device pointer
# @param [in] srcDeviceId - Source device
# @param [in] sizeBytes - Size of memory copy in bytes
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
cdef hipError_t hipMemcpyPeer(void * dst,int dstDeviceId,const void * src,int srcDeviceId,int sizeBytes) nogil:
    global _lib_handle
    global hipMemcpyPeer_funptr
    if hipMemcpyPeer_funptr == NULL:
        with gil:
            hipMemcpyPeer_funptr = loader.load_symbol(_lib_handle, "hipMemcpyPeer")
    return (<hipError_t (*)(void *,int,const void *,int,int) nogil> hipMemcpyPeer_funptr)(dst,dstDeviceId,src,srcDeviceId,sizeBytes)


cdef void* hipMemcpyPeerAsync_funptr = NULL
# @brief Copies memory from one device to memory on another device.
# @param [out] dst - Destination device pointer.
# @param [in] dstDevice - Destination device
# @param [in] src - Source device pointer
# @param [in] srcDevice - Source device
# @param [in] sizeBytes - Size of memory copy in bytes
# @param [in] stream - Stream identifier
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
cdef hipError_t hipMemcpyPeerAsync(void * dst,int dstDeviceId,const void * src,int srcDevice,int sizeBytes,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyPeerAsync_funptr
    if hipMemcpyPeerAsync_funptr == NULL:
        with gil:
            hipMemcpyPeerAsync_funptr = loader.load_symbol(_lib_handle, "hipMemcpyPeerAsync")
    return (<hipError_t (*)(void *,int,const void *,int,int,hipStream_t) nogil> hipMemcpyPeerAsync_funptr)(dst,dstDeviceId,src,srcDevice,sizeBytes,stream)


cdef void* hipCtxCreate_funptr = NULL
# @}
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# @defgroup Context Context Management
# @{
# This section describes the context management functions of HIP runtime API.
# @addtogroup ContextD Context Management [Deprecated]
# @{
# @ingroup Context
# This section describes the deprecated context management functions of HIP runtime API.
# @brief Create a context and set it as current/ default context
# @param [out] ctx
# @param [in] flags
# @param [in] associated device handle
# @return #hipSuccess
# @see hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent,
# hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxCreate(hipCtx_t* ctx,unsigned int flags,hipDevice_t device) nogil:
    global _lib_handle
    global hipCtxCreate_funptr
    if hipCtxCreate_funptr == NULL:
        with gil:
            hipCtxCreate_funptr = loader.load_symbol(_lib_handle, "hipCtxCreate")
    return (<hipError_t (*)(hipCtx_t*,unsigned int,hipDevice_t) nogil> hipCtxCreate_funptr)(ctx,flags,device)


cdef void* hipCtxDestroy_funptr = NULL
# @brief Destroy a HIP context.
# @param [in] ctx Context to destroy
# @returns #hipSuccess, #hipErrorInvalidValue
# @see hipCtxCreate, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,hipCtxSetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
cdef hipError_t hipCtxDestroy(hipCtx_t ctx) nogil:
    global _lib_handle
    global hipCtxDestroy_funptr
    if hipCtxDestroy_funptr == NULL:
        with gil:
            hipCtxDestroy_funptr = loader.load_symbol(_lib_handle, "hipCtxDestroy")
    return (<hipError_t (*)(hipCtx_t) nogil> hipCtxDestroy_funptr)(ctx)


cdef void* hipCtxPopCurrent_funptr = NULL
# @brief Pop the current/default context and return the popped context.
# @param [out] ctx
# @returns #hipSuccess, #hipErrorInvalidContext
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxSetCurrent, hipCtxGetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxPopCurrent(hipCtx_t* ctx) nogil:
    global _lib_handle
    global hipCtxPopCurrent_funptr
    if hipCtxPopCurrent_funptr == NULL:
        with gil:
            hipCtxPopCurrent_funptr = loader.load_symbol(_lib_handle, "hipCtxPopCurrent")
    return (<hipError_t (*)(hipCtx_t*) nogil> hipCtxPopCurrent_funptr)(ctx)


cdef void* hipCtxPushCurrent_funptr = NULL
# @brief Push the context to be set as current/ default context
# @param [in] ctx
# @returns #hipSuccess, #hipErrorInvalidContext
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
cdef hipError_t hipCtxPushCurrent(hipCtx_t ctx) nogil:
    global _lib_handle
    global hipCtxPushCurrent_funptr
    if hipCtxPushCurrent_funptr == NULL:
        with gil:
            hipCtxPushCurrent_funptr = loader.load_symbol(_lib_handle, "hipCtxPushCurrent")
    return (<hipError_t (*)(hipCtx_t) nogil> hipCtxPushCurrent_funptr)(ctx)


cdef void* hipCtxSetCurrent_funptr = NULL
# @brief Set the passed context as current/default
# @param [in] ctx
# @returns #hipSuccess, #hipErrorInvalidContext
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
cdef hipError_t hipCtxSetCurrent(hipCtx_t ctx) nogil:
    global _lib_handle
    global hipCtxSetCurrent_funptr
    if hipCtxSetCurrent_funptr == NULL:
        with gil:
            hipCtxSetCurrent_funptr = loader.load_symbol(_lib_handle, "hipCtxSetCurrent")
    return (<hipError_t (*)(hipCtx_t) nogil> hipCtxSetCurrent_funptr)(ctx)


cdef void* hipCtxGetCurrent_funptr = NULL
# @brief Get the handle of the current/ default context
# @param [out] ctx
# @returns #hipSuccess, #hipErrorInvalidContext
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetCurrent(hipCtx_t* ctx) nogil:
    global _lib_handle
    global hipCtxGetCurrent_funptr
    if hipCtxGetCurrent_funptr == NULL:
        with gil:
            hipCtxGetCurrent_funptr = loader.load_symbol(_lib_handle, "hipCtxGetCurrent")
    return (<hipError_t (*)(hipCtx_t*) nogil> hipCtxGetCurrent_funptr)(ctx)


cdef void* hipCtxGetDevice_funptr = NULL
# @brief Get the handle of the device associated with current/default context
# @param [out] device
# @returns #hipSuccess, #hipErrorInvalidContext
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize
cdef hipError_t hipCtxGetDevice(hipDevice_t * device) nogil:
    global _lib_handle
    global hipCtxGetDevice_funptr
    if hipCtxGetDevice_funptr == NULL:
        with gil:
            hipCtxGetDevice_funptr = loader.load_symbol(_lib_handle, "hipCtxGetDevice")
    return (<hipError_t (*)(hipDevice_t *) nogil> hipCtxGetDevice_funptr)(device)


cdef void* hipCtxGetApiVersion_funptr = NULL
# @brief Returns the approximate HIP api version.
# @param [in]  ctx Context to check
# @param [out] apiVersion
# @return #hipSuccess
# @warning The HIP feature set does not correspond to an exact CUDA SDK api revision.
# This function always set *apiVersion to 4 as an approximation though HIP supports
# some features which were introduced in later CUDA SDK revisions.
# HIP apps code should not rely on the api revision number here and should
# use arch feature flags to test device capabilities or conditional compilation.
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetApiVersion(hipCtx_t ctx,int * apiVersion) nogil:
    global _lib_handle
    global hipCtxGetApiVersion_funptr
    if hipCtxGetApiVersion_funptr == NULL:
        with gil:
            hipCtxGetApiVersion_funptr = loader.load_symbol(_lib_handle, "hipCtxGetApiVersion")
    return (<hipError_t (*)(hipCtx_t,int *) nogil> hipCtxGetApiVersion_funptr)(ctx,apiVersion)


cdef void* hipCtxGetCacheConfig_funptr = NULL
# @brief Set Cache configuration for a specific function
# @param [out] cacheConfiguration
# @return #hipSuccess
# @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
# ignored on those architectures.
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetCacheConfig(hipFuncCache_t * cacheConfig) nogil:
    global _lib_handle
    global hipCtxGetCacheConfig_funptr
    if hipCtxGetCacheConfig_funptr == NULL:
        with gil:
            hipCtxGetCacheConfig_funptr = loader.load_symbol(_lib_handle, "hipCtxGetCacheConfig")
    return (<hipError_t (*)(hipFuncCache_t *) nogil> hipCtxGetCacheConfig_funptr)(cacheConfig)


cdef void* hipCtxSetCacheConfig_funptr = NULL
# @brief Set L1/Shared cache partition.
# @param [in] cacheConfiguration
# @return #hipSuccess
# @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
# ignored on those architectures.
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) nogil:
    global _lib_handle
    global hipCtxSetCacheConfig_funptr
    if hipCtxSetCacheConfig_funptr == NULL:
        with gil:
            hipCtxSetCacheConfig_funptr = loader.load_symbol(_lib_handle, "hipCtxSetCacheConfig")
    return (<hipError_t (*)(hipFuncCache_t) nogil> hipCtxSetCacheConfig_funptr)(cacheConfig)


cdef void* hipCtxSetSharedMemConfig_funptr = NULL
# @brief Set Shared memory bank configuration.
# @param [in] sharedMemoryConfiguration
# @return #hipSuccess
# @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) nogil:
    global _lib_handle
    global hipCtxSetSharedMemConfig_funptr
    if hipCtxSetSharedMemConfig_funptr == NULL:
        with gil:
            hipCtxSetSharedMemConfig_funptr = loader.load_symbol(_lib_handle, "hipCtxSetSharedMemConfig")
    return (<hipError_t (*)(hipSharedMemConfig) nogil> hipCtxSetSharedMemConfig_funptr)(config)


cdef void* hipCtxGetSharedMemConfig_funptr = NULL
# @brief Get Shared memory bank configuration.
# @param [out] sharedMemoryConfiguration
# @return #hipSuccess
# @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig * pConfig) nogil:
    global _lib_handle
    global hipCtxGetSharedMemConfig_funptr
    if hipCtxGetSharedMemConfig_funptr == NULL:
        with gil:
            hipCtxGetSharedMemConfig_funptr = loader.load_symbol(_lib_handle, "hipCtxGetSharedMemConfig")
    return (<hipError_t (*)(hipSharedMemConfig *) nogil> hipCtxGetSharedMemConfig_funptr)(pConfig)


cdef void* hipCtxSynchronize_funptr = NULL
# @brief Blocks until the default context has completed all preceding requested tasks.
# @return #hipSuccess
# @warning This function waits for all streams on the default context to complete execution, and
# then returns.
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxGetDevice
cdef hipError_t hipCtxSynchronize() nogil:
    global _lib_handle
    global hipCtxSynchronize_funptr
    if hipCtxSynchronize_funptr == NULL:
        with gil:
            hipCtxSynchronize_funptr = loader.load_symbol(_lib_handle, "hipCtxSynchronize")
    return (<hipError_t (*)() nogil> hipCtxSynchronize_funptr)()


cdef void* hipCtxGetFlags_funptr = NULL
# @brief Return flags used for creating default context.
# @param [out] flags
# @returns #hipSuccess
# @see hipCtxCreate, hipCtxDestroy, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetFlags(unsigned int * flags) nogil:
    global _lib_handle
    global hipCtxGetFlags_funptr
    if hipCtxGetFlags_funptr == NULL:
        with gil:
            hipCtxGetFlags_funptr = loader.load_symbol(_lib_handle, "hipCtxGetFlags")
    return (<hipError_t (*)(unsigned int *) nogil> hipCtxGetFlags_funptr)(flags)


cdef void* hipCtxEnablePeerAccess_funptr = NULL
# @brief Enables direct access to memory allocations in a peer context.
# Memory which already allocated on peer device will be mapped into the address space of the
# current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
# the address space of the current device when the memory is allocated. The peer memory remains
# accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
# @param [in] peerCtx
# @param [in] flags
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
# #hipErrorPeerAccessAlreadyEnabled
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
# @warning PeerToPeer support is experimental.
cdef hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx,unsigned int flags) nogil:
    global _lib_handle
    global hipCtxEnablePeerAccess_funptr
    if hipCtxEnablePeerAccess_funptr == NULL:
        with gil:
            hipCtxEnablePeerAccess_funptr = loader.load_symbol(_lib_handle, "hipCtxEnablePeerAccess")
    return (<hipError_t (*)(hipCtx_t,unsigned int) nogil> hipCtxEnablePeerAccess_funptr)(peerCtx,flags)


cdef void* hipCtxDisablePeerAccess_funptr = NULL
# @brief Disable direct access from current context's virtual address space to memory allocations
# physically located on a peer context.Disables direct access to memory allocations in a peer
# context and unregisters any registered allocations.
# Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
# enabled from the current device.
# @param [in] peerCtx
# @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
# @warning PeerToPeer support is experimental.
cdef hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) nogil:
    global _lib_handle
    global hipCtxDisablePeerAccess_funptr
    if hipCtxDisablePeerAccess_funptr == NULL:
        with gil:
            hipCtxDisablePeerAccess_funptr = loader.load_symbol(_lib_handle, "hipCtxDisablePeerAccess")
    return (<hipError_t (*)(hipCtx_t) nogil> hipCtxDisablePeerAccess_funptr)(peerCtx)


cdef void* hipDevicePrimaryCtxGetState_funptr = NULL
# @}
# @brief Get the state of the primary context.
# @param [in] Device to get primary context flags for
# @param [out] Pointer to store flags
# @param [out] Pointer to store context state; 0 = inactive, 1 = active
# @returns #hipSuccess
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev,unsigned int * flags,int * active) nogil:
    global _lib_handle
    global hipDevicePrimaryCtxGetState_funptr
    if hipDevicePrimaryCtxGetState_funptr == NULL:
        with gil:
            hipDevicePrimaryCtxGetState_funptr = loader.load_symbol(_lib_handle, "hipDevicePrimaryCtxGetState")
    return (<hipError_t (*)(hipDevice_t,unsigned int *,int *) nogil> hipDevicePrimaryCtxGetState_funptr)(dev,flags,active)


cdef void* hipDevicePrimaryCtxRelease_funptr = NULL
# @brief Release the primary context on the GPU.
# @param [in] Device which primary context is released
# @returns #hipSuccess
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
# @warning This function return #hipSuccess though doesn't release the primaryCtx by design on
# HIP/HCC path.
cdef hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) nogil:
    global _lib_handle
    global hipDevicePrimaryCtxRelease_funptr
    if hipDevicePrimaryCtxRelease_funptr == NULL:
        with gil:
            hipDevicePrimaryCtxRelease_funptr = loader.load_symbol(_lib_handle, "hipDevicePrimaryCtxRelease")
    return (<hipError_t (*)(hipDevice_t) nogil> hipDevicePrimaryCtxRelease_funptr)(dev)


cdef void* hipDevicePrimaryCtxRetain_funptr = NULL
# @brief Retain the primary context on the GPU.
# @param [out] Returned context handle of the new context
# @param [in] Device which primary context is released
# @returns #hipSuccess
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx,hipDevice_t dev) nogil:
    global _lib_handle
    global hipDevicePrimaryCtxRetain_funptr
    if hipDevicePrimaryCtxRetain_funptr == NULL:
        with gil:
            hipDevicePrimaryCtxRetain_funptr = loader.load_symbol(_lib_handle, "hipDevicePrimaryCtxRetain")
    return (<hipError_t (*)(hipCtx_t*,hipDevice_t) nogil> hipDevicePrimaryCtxRetain_funptr)(pctx,dev)


cdef void* hipDevicePrimaryCtxReset_funptr = NULL
# @brief Resets the primary context on the GPU.
# @param [in] Device which primary context is reset
# @returns #hipSuccess
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) nogil:
    global _lib_handle
    global hipDevicePrimaryCtxReset_funptr
    if hipDevicePrimaryCtxReset_funptr == NULL:
        with gil:
            hipDevicePrimaryCtxReset_funptr = loader.load_symbol(_lib_handle, "hipDevicePrimaryCtxReset")
    return (<hipError_t (*)(hipDevice_t) nogil> hipDevicePrimaryCtxReset_funptr)(dev)


cdef void* hipDevicePrimaryCtxSetFlags_funptr = NULL
# @brief Set flags for the primary context.
# @param [in] Device for which the primary context flags are set
# @param [in] New flags for the device
# @returns #hipSuccess, #hipErrorContextAlreadyInUse
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev,unsigned int flags) nogil:
    global _lib_handle
    global hipDevicePrimaryCtxSetFlags_funptr
    if hipDevicePrimaryCtxSetFlags_funptr == NULL:
        with gil:
            hipDevicePrimaryCtxSetFlags_funptr = loader.load_symbol(_lib_handle, "hipDevicePrimaryCtxSetFlags")
    return (<hipError_t (*)(hipDevice_t,unsigned int) nogil> hipDevicePrimaryCtxSetFlags_funptr)(dev,flags)


cdef void* hipModuleLoad_funptr = NULL
# @}
# @defgroup Module Module Management
# @{
# This section describes the module management functions of HIP runtime API.
# @brief Loads code object from file into a hipModule_t
# @param [in] fname
# @param [out] module
# @warning File/memory resources allocated in this function are released only in hipModuleUnload.
# @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorFileNotFound,
# hipErrorOutOfMemory, hipErrorSharedObjectInitFailed, hipErrorNotInitialized
cdef hipError_t hipModuleLoad(hipModule_t* module,const char * fname) nogil:
    global _lib_handle
    global hipModuleLoad_funptr
    if hipModuleLoad_funptr == NULL:
        with gil:
            hipModuleLoad_funptr = loader.load_symbol(_lib_handle, "hipModuleLoad")
    return (<hipError_t (*)(hipModule_t*,const char *) nogil> hipModuleLoad_funptr)(module,fname)


cdef void* hipModuleUnload_funptr = NULL
# @brief Frees the module
# @param [in] module
# @returns hipSuccess, hipInvalidValue
# module is freed and the code objects associated with it are destroyed
cdef hipError_t hipModuleUnload(hipModule_t module) nogil:
    global _lib_handle
    global hipModuleUnload_funptr
    if hipModuleUnload_funptr == NULL:
        with gil:
            hipModuleUnload_funptr = loader.load_symbol(_lib_handle, "hipModuleUnload")
    return (<hipError_t (*)(hipModule_t) nogil> hipModuleUnload_funptr)(module)


cdef void* hipModuleGetFunction_funptr = NULL
# @brief Function with kname will be extracted if present in module
# @param [in] module
# @param [in] kname
# @param [out] function
# @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorNotInitialized,
# hipErrorNotFound,
cdef hipError_t hipModuleGetFunction(hipFunction_t* function,hipModule_t module,const char * kname) nogil:
    global _lib_handle
    global hipModuleGetFunction_funptr
    if hipModuleGetFunction_funptr == NULL:
        with gil:
            hipModuleGetFunction_funptr = loader.load_symbol(_lib_handle, "hipModuleGetFunction")
    return (<hipError_t (*)(hipFunction_t*,hipModule_t,const char *) nogil> hipModuleGetFunction_funptr)(function,module,kname)


cdef void* hipFuncGetAttributes_funptr = NULL
# @brief Find out attributes for a given function.
# @param [out] attr
# @param [in] func
# @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
cdef hipError_t hipFuncGetAttributes(hipFuncAttributes * attr,const void * func) nogil:
    global _lib_handle
    global hipFuncGetAttributes_funptr
    if hipFuncGetAttributes_funptr == NULL:
        with gil:
            hipFuncGetAttributes_funptr = loader.load_symbol(_lib_handle, "hipFuncGetAttributes")
    return (<hipError_t (*)(hipFuncAttributes *,const void *) nogil> hipFuncGetAttributes_funptr)(attr,func)


cdef void* hipFuncGetAttribute_funptr = NULL
# @brief Find out a specific attribute for a given function.
# @param [out] value
# @param [in]  attrib
# @param [in]  hfunc
# @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
cdef hipError_t hipFuncGetAttribute(int * value,hipFunction_attribute attrib,hipFunction_t hfunc) nogil:
    global _lib_handle
    global hipFuncGetAttribute_funptr
    if hipFuncGetAttribute_funptr == NULL:
        with gil:
            hipFuncGetAttribute_funptr = loader.load_symbol(_lib_handle, "hipFuncGetAttribute")
    return (<hipError_t (*)(int *,hipFunction_attribute,hipFunction_t) nogil> hipFuncGetAttribute_funptr)(value,attrib,hfunc)


cdef void* hipModuleGetTexRef_funptr = NULL
# @brief returns the handle of the texture reference with the name from the module.
# @param [in] hmod
# @param [in] name
# @param [out] texRef
# @returns hipSuccess, hipErrorNotInitialized, hipErrorNotFound, hipErrorInvalidValue
cdef hipError_t hipModuleGetTexRef(textureReference ** texRef,hipModule_t hmod,const char * name) nogil:
    global _lib_handle
    global hipModuleGetTexRef_funptr
    if hipModuleGetTexRef_funptr == NULL:
        with gil:
            hipModuleGetTexRef_funptr = loader.load_symbol(_lib_handle, "hipModuleGetTexRef")
    return (<hipError_t (*)(textureReference **,hipModule_t,const char *) nogil> hipModuleGetTexRef_funptr)(texRef,hmod,name)


cdef void* hipModuleLoadData_funptr = NULL
# @brief builds module from code object which resides in host memory. Image is pointer to that
# location.
# @param [in] image
# @param [out] module
# @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
cdef hipError_t hipModuleLoadData(hipModule_t* module,const void * image) nogil:
    global _lib_handle
    global hipModuleLoadData_funptr
    if hipModuleLoadData_funptr == NULL:
        with gil:
            hipModuleLoadData_funptr = loader.load_symbol(_lib_handle, "hipModuleLoadData")
    return (<hipError_t (*)(hipModule_t*,const void *) nogil> hipModuleLoadData_funptr)(module,image)


cdef void* hipModuleLoadDataEx_funptr = NULL
# @brief builds module from code object which resides in host memory. Image is pointer to that
# location. Options are not used. hipModuleLoadData is called.
# @param [in] image
# @param [out] module
# @param [in] number of options
# @param [in] options for JIT
# @param [in] option values for JIT
# @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
cdef hipError_t hipModuleLoadDataEx(hipModule_t* module,const void * image,unsigned int numOptions,hipJitOption * options,void ** optionValues) nogil:
    global _lib_handle
    global hipModuleLoadDataEx_funptr
    if hipModuleLoadDataEx_funptr == NULL:
        with gil:
            hipModuleLoadDataEx_funptr = loader.load_symbol(_lib_handle, "hipModuleLoadDataEx")
    return (<hipError_t (*)(hipModule_t*,const void *,unsigned int,hipJitOption *,void **) nogil> hipModuleLoadDataEx_funptr)(module,image,numOptions,options,optionValues)


cdef void* hipModuleLaunchKernel_funptr = NULL
# @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
# to kernelparams or extra
# @param [in] f         Kernel to launch.
# @param [in] gridDimX  X grid dimension specified as multiple of blockDimX.
# @param [in] gridDimY  Y grid dimension specified as multiple of blockDimY.
# @param [in] gridDimZ  Z grid dimension specified as multiple of blockDimZ.
# @param [in] blockDimX X block dimensions specified in work-items
# @param [in] blockDimY Y grid dimension specified in work-items
# @param [in] blockDimZ Z grid dimension specified in work-items
# @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
# HIP-Clang compiler provides support for extern shared declarations.
# @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
# default stream is used with associated synchronization rules.
# @param [in] kernelParams
# @param [in] extra     Pointer to kernel arguments.   These are passed directly to the kernel and
# must be in the memory layout and alignment expected by the kernel.
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32. So gridDim.x * blockDim.x, gridDim.y * blockDim.y
# and gridDim.z * blockDim.z are always less than 2^32.
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
# @warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please
# refer to hip_porting_driver_api.md for sample usage.
cdef hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,hipStream_t stream,void ** kernelParams,void ** extra) nogil:
    global _lib_handle
    global hipModuleLaunchKernel_funptr
    if hipModuleLaunchKernel_funptr == NULL:
        with gil:
            hipModuleLaunchKernel_funptr = loader.load_symbol(_lib_handle, "hipModuleLaunchKernel")
    return (<hipError_t (*)(hipFunction_t,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,hipStream_t,void **,void **) nogil> hipModuleLaunchKernel_funptr)(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,stream,kernelParams,extra)


cdef void* hipLaunchCooperativeKernel_funptr = NULL
# @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
# to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute
# @param [in] f         Kernel to launch.
# @param [in] gridDim   Grid dimensions specified as multiple of blockDim.
# @param [in] blockDim  Block dimensions specified in work-items
# @param [in] kernelParams A list of kernel arguments
# @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
# HIP-Clang compiler provides support for extern shared declarations.
# @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
# default stream is used with associated synchronization rules.
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32.
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
cdef hipError_t hipLaunchCooperativeKernel(const void * f,dim3 gridDim,dim3 blockDimX,void ** kernelParams,unsigned int sharedMemBytes,hipStream_t stream) nogil:
    global _lib_handle
    global hipLaunchCooperativeKernel_funptr
    if hipLaunchCooperativeKernel_funptr == NULL:
        with gil:
            hipLaunchCooperativeKernel_funptr = loader.load_symbol(_lib_handle, "hipLaunchCooperativeKernel")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,unsigned int,hipStream_t) nogil> hipLaunchCooperativeKernel_funptr)(f,gridDim,blockDimX,kernelParams,sharedMemBytes,stream)


cdef void* hipLaunchCooperativeKernelMultiDevice_funptr = NULL
# @brief Launches kernels on multiple devices where thread blocks can cooperate and
# synchronize as they execute.
# @param [in] launchParamsList         List of launch parameters, one per device.
# @param [in] numDevices               Size of the launchParamsList array.
# @param [in] flags                    Flags to control launch behavior.
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
cdef hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams_t * launchParamsList,int numDevices,unsigned int flags) nogil:
    global _lib_handle
    global hipLaunchCooperativeKernelMultiDevice_funptr
    if hipLaunchCooperativeKernelMultiDevice_funptr == NULL:
        with gil:
            hipLaunchCooperativeKernelMultiDevice_funptr = loader.load_symbol(_lib_handle, "hipLaunchCooperativeKernelMultiDevice")
    return (<hipError_t (*)(hipLaunchParams_t *,int,unsigned int) nogil> hipLaunchCooperativeKernelMultiDevice_funptr)(launchParamsList,numDevices,flags)


cdef void* hipExtLaunchMultiKernelMultiDevice_funptr = NULL
# @brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
# on respective streams before enqueuing any other work on the specified streams from any other threads
# @param [in] hipLaunchParams          List of launch parameters, one per device.
# @param [in] numDevices               Size of the launchParamsList array.
# @param [in] flags                    Flags to control launch behavior.
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
cdef hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams_t * launchParamsList,int numDevices,unsigned int flags) nogil:
    global _lib_handle
    global hipExtLaunchMultiKernelMultiDevice_funptr
    if hipExtLaunchMultiKernelMultiDevice_funptr == NULL:
        with gil:
            hipExtLaunchMultiKernelMultiDevice_funptr = loader.load_symbol(_lib_handle, "hipExtLaunchMultiKernelMultiDevice")
    return (<hipError_t (*)(hipLaunchParams_t *,int,unsigned int) nogil> hipExtLaunchMultiKernelMultiDevice_funptr)(launchParamsList,numDevices,flags)


cdef void* hipModuleOccupancyMaxPotentialBlockSize_funptr = NULL
# @}
# @defgroup Occupancy Occupancy
# @{
# This section describes the occupancy functions of HIP runtime API.
# @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
# @param [out] gridSize           minimum grid size for maximum potential occupancy
# @param [out] blockSize          block size for maximum potential occupancy
# @param [in]  f                  kernel function for which occupancy is calulated
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32.
# @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
cdef hipError_t hipModuleOccupancyMaxPotentialBlockSize(int * gridSize,int * blockSize,hipFunction_t f,int dynSharedMemPerBlk,int blockSizeLimit) nogil:
    global _lib_handle
    global hipModuleOccupancyMaxPotentialBlockSize_funptr
    if hipModuleOccupancyMaxPotentialBlockSize_funptr == NULL:
        with gil:
            hipModuleOccupancyMaxPotentialBlockSize_funptr = loader.load_symbol(_lib_handle, "hipModuleOccupancyMaxPotentialBlockSize")
    return (<hipError_t (*)(int *,int *,hipFunction_t,int,int) nogil> hipModuleOccupancyMaxPotentialBlockSize_funptr)(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit)


cdef void* hipModuleOccupancyMaxPotentialBlockSizeWithFlags_funptr = NULL
# @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
# @param [out] gridSize           minimum grid size for maximum potential occupancy
# @param [out] blockSize          block size for maximum potential occupancy
# @param [in]  f                  kernel function for which occupancy is calulated
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
# @param [in]  flags            Extra flags for occupancy calculation (only default supported)
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32.
# @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
cdef hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int * gridSize,int * blockSize,hipFunction_t f,int dynSharedMemPerBlk,int blockSizeLimit,unsigned int flags) nogil:
    global _lib_handle
    global hipModuleOccupancyMaxPotentialBlockSizeWithFlags_funptr
    if hipModuleOccupancyMaxPotentialBlockSizeWithFlags_funptr == NULL:
        with gil:
            hipModuleOccupancyMaxPotentialBlockSizeWithFlags_funptr = loader.load_symbol(_lib_handle, "hipModuleOccupancyMaxPotentialBlockSizeWithFlags")
    return (<hipError_t (*)(int *,int *,hipFunction_t,int,int,unsigned int) nogil> hipModuleOccupancyMaxPotentialBlockSizeWithFlags_funptr)(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit,flags)


cdef void* hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_funptr = NULL
# @brief Returns occupancy for a device function.
# @param [out] numBlocks        Returned occupancy
# @param [in]  func             Kernel function (hipFunction) for which occupancy is calulated
# @param [in]  blockSize        Block size the kernel is intended to be launched with
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
cdef hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,hipFunction_t f,int blockSize,int dynSharedMemPerBlk) nogil:
    global _lib_handle
    global hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_funptr
    if hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_funptr == NULL:
        with gil:
            hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_funptr = loader.load_symbol(_lib_handle, "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor")
    return (<hipError_t (*)(int *,hipFunction_t,int,int) nogil> hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_funptr)(numBlocks,f,blockSize,dynSharedMemPerBlk)


cdef void* hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr = NULL
# @brief Returns occupancy for a device function.
# @param [out] numBlocks        Returned occupancy
# @param [in]  f                Kernel function(hipFunction_t) for which occupancy is calulated
# @param [in]  blockSize        Block size the kernel is intended to be launched with
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  flags            Extra flags for occupancy calculation (only default supported)
cdef hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,hipFunction_t f,int blockSize,int dynSharedMemPerBlk,unsigned int flags) nogil:
    global _lib_handle
    global hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr
    if hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr == NULL:
        with gil:
            hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr = loader.load_symbol(_lib_handle, "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")
    return (<hipError_t (*)(int *,hipFunction_t,int,int,unsigned int) nogil> hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr)(numBlocks,f,blockSize,dynSharedMemPerBlk,flags)


cdef void* hipOccupancyMaxActiveBlocksPerMultiprocessor_funptr = NULL
# @brief Returns occupancy for a device function.
# @param [out] numBlocks        Returned occupancy
# @param [in]  func             Kernel function for which occupancy is calulated
# @param [in]  blockSize        Block size the kernel is intended to be launched with
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
cdef hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,const void * f,int blockSize,int dynSharedMemPerBlk) nogil:
    global _lib_handle
    global hipOccupancyMaxActiveBlocksPerMultiprocessor_funptr
    if hipOccupancyMaxActiveBlocksPerMultiprocessor_funptr == NULL:
        with gil:
            hipOccupancyMaxActiveBlocksPerMultiprocessor_funptr = loader.load_symbol(_lib_handle, "hipOccupancyMaxActiveBlocksPerMultiprocessor")
    return (<hipError_t (*)(int *,const void *,int,int) nogil> hipOccupancyMaxActiveBlocksPerMultiprocessor_funptr)(numBlocks,f,blockSize,dynSharedMemPerBlk)


cdef void* hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr = NULL
# @brief Returns occupancy for a device function.
# @param [out] numBlocks        Returned occupancy
# @param [in]  f                Kernel function for which occupancy is calulated
# @param [in]  blockSize        Block size the kernel is intended to be launched with
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  flags            Extra flags for occupancy calculation (currently ignored)
cdef hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,const void * f,int blockSize,int dynSharedMemPerBlk,unsigned int flags) nogil:
    global _lib_handle
    global hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr
    if hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr == NULL:
        with gil:
            hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr = loader.load_symbol(_lib_handle, "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")
    return (<hipError_t (*)(int *,const void *,int,int,unsigned int) nogil> hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_funptr)(numBlocks,f,blockSize,dynSharedMemPerBlk,flags)


cdef void* hipOccupancyMaxPotentialBlockSize_funptr = NULL
# @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
# @param [out] gridSize           minimum grid size for maximum potential occupancy
# @param [out] blockSize          block size for maximum potential occupancy
# @param [in]  f                  kernel function for which occupancy is calulated
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32.
# @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
cdef hipError_t hipOccupancyMaxPotentialBlockSize(int * gridSize,int * blockSize,const void * f,int dynSharedMemPerBlk,int blockSizeLimit) nogil:
    global _lib_handle
    global hipOccupancyMaxPotentialBlockSize_funptr
    if hipOccupancyMaxPotentialBlockSize_funptr == NULL:
        with gil:
            hipOccupancyMaxPotentialBlockSize_funptr = loader.load_symbol(_lib_handle, "hipOccupancyMaxPotentialBlockSize")
    return (<hipError_t (*)(int *,int *,const void *,int,int) nogil> hipOccupancyMaxPotentialBlockSize_funptr)(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit)


cdef void* hipProfilerStart_funptr = NULL
# @brief Start recording of profiling information
# When using this API, start the profiler with profiling disabled.  (--startdisabled)
# @warning : hipProfilerStart API is under development.
cdef hipError_t hipProfilerStart() nogil:
    global _lib_handle
    global hipProfilerStart_funptr
    if hipProfilerStart_funptr == NULL:
        with gil:
            hipProfilerStart_funptr = loader.load_symbol(_lib_handle, "hipProfilerStart")
    return (<hipError_t (*)() nogil> hipProfilerStart_funptr)()


cdef void* hipProfilerStop_funptr = NULL
# @brief Stop recording of profiling information.
# When using this API, start the profiler with profiling disabled.  (--startdisabled)
# @warning : hipProfilerStop API is under development.
cdef hipError_t hipProfilerStop() nogil:
    global _lib_handle
    global hipProfilerStop_funptr
    if hipProfilerStop_funptr == NULL:
        with gil:
            hipProfilerStop_funptr = loader.load_symbol(_lib_handle, "hipProfilerStop")
    return (<hipError_t (*)() nogil> hipProfilerStop_funptr)()


cdef void* hipConfigureCall_funptr = NULL
# @}
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# @defgroup Clang Launch API to support the triple-chevron syntax
# @{
# This section describes the API to support the triple-chevron syntax.
# @brief Configure a kernel launch.
# @param [in] gridDim   grid dimension specified as multiple of blockDim.
# @param [in] blockDim  block dimensions specified in work-items
# @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
# HIP-Clang compiler provides support for extern shared declarations.
# @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
# default stream is used with associated synchronization rules.
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32.
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
cdef hipError_t hipConfigureCall(dim3 gridDim,dim3 blockDim,int sharedMem,hipStream_t stream) nogil:
    global _lib_handle
    global hipConfigureCall_funptr
    if hipConfigureCall_funptr == NULL:
        with gil:
            hipConfigureCall_funptr = loader.load_symbol(_lib_handle, "hipConfigureCall")
    return (<hipError_t (*)(dim3,dim3,int,hipStream_t) nogil> hipConfigureCall_funptr)(gridDim,blockDim,sharedMem,stream)


cdef void* hipSetupArgument_funptr = NULL
# @brief Set a kernel argument.
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
# @param [in] arg    Pointer the argument in host memory.
# @param [in] size   Size of the argument.
# @param [in] offset Offset of the argument on the argument stack.
cdef hipError_t hipSetupArgument(const void * arg,int size,int offset) nogil:
    global _lib_handle
    global hipSetupArgument_funptr
    if hipSetupArgument_funptr == NULL:
        with gil:
            hipSetupArgument_funptr = loader.load_symbol(_lib_handle, "hipSetupArgument")
    return (<hipError_t (*)(const void *,int,int) nogil> hipSetupArgument_funptr)(arg,size,offset)


cdef void* hipLaunchByPtr_funptr = NULL
# @brief Launch a kernel.
# @param [in] func Kernel to launch.
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
cdef hipError_t hipLaunchByPtr(const void * func) nogil:
    global _lib_handle
    global hipLaunchByPtr_funptr
    if hipLaunchByPtr_funptr == NULL:
        with gil:
            hipLaunchByPtr_funptr = loader.load_symbol(_lib_handle, "hipLaunchByPtr")
    return (<hipError_t (*)(const void *) nogil> hipLaunchByPtr_funptr)(func)


cdef void* hipLaunchKernel_funptr = NULL
# @brief C compliant kernel launch API
# @param [in] function_address - kernel stub function pointer.
# @param [in] numBlocks - number of blocks
# @param [in] dimBlocks - dimension of a block
# @param [in] args - kernel arguments
# @param [in] sharedMemBytes - Amount of dynamic shared memory to allocate for this kernel. The
# HIP-Clang compiler provides support for extern shared declarations.
# @param [in] stream - Stream where the kernel should be dispatched.  May be 0, in which case th
# default stream is used with associated synchronization rules.
# @returns #hipSuccess, #hipErrorInvalidValue, hipInvalidDevice
cdef hipError_t hipLaunchKernel(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,int sharedMemBytes,hipStream_t stream) nogil:
    global _lib_handle
    global hipLaunchKernel_funptr
    if hipLaunchKernel_funptr == NULL:
        with gil:
            hipLaunchKernel_funptr = loader.load_symbol(_lib_handle, "hipLaunchKernel")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,int,hipStream_t) nogil> hipLaunchKernel_funptr)(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream)


cdef void* hipLaunchHostFunc_funptr = NULL
# @brief Enqueues a host function call in a stream.
# @param [in] stream - stream to enqueue work to.
# @param [in] fn - function to call once operations enqueued preceeding are complete.
# @param [in] userData - User-specified data to be passed to the function.
# @returns #hipSuccess, #hipErrorInvalidResourceHandle, #hipErrorInvalidValue,
# #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipLaunchHostFunc(hipStream_t stream,hipHostFn_t fn,void * userData) nogil:
    global _lib_handle
    global hipLaunchHostFunc_funptr
    if hipLaunchHostFunc_funptr == NULL:
        with gil:
            hipLaunchHostFunc_funptr = loader.load_symbol(_lib_handle, "hipLaunchHostFunc")
    return (<hipError_t (*)(hipStream_t,hipHostFn_t,void *) nogil> hipLaunchHostFunc_funptr)(stream,fn,userData)


cdef void* hipDrvMemcpy2DUnaligned_funptr = NULL
# Copies memory for 2D arrays.
# @param pCopy           - Parameters for the memory copy
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipDrvMemcpy2DUnaligned(hip_Memcpy2D * pCopy) nogil:
    global _lib_handle
    global hipDrvMemcpy2DUnaligned_funptr
    if hipDrvMemcpy2DUnaligned_funptr == NULL:
        with gil:
            hipDrvMemcpy2DUnaligned_funptr = loader.load_symbol(_lib_handle, "hipDrvMemcpy2DUnaligned")
    return (<hipError_t (*)(hip_Memcpy2D *) nogil> hipDrvMemcpy2DUnaligned_funptr)(pCopy)


cdef void* hipExtLaunchKernel_funptr = NULL
# @brief Launches kernel from the pointer address, with arguments and shared memory on stream.
# @param [in] function_address pointer to the Kernel to launch.
# @param [in] numBlocks number of blocks.
# @param [in] dimBlocks dimension of a block.
# @param [in] args pointer to kernel arguments.
# @param [in] sharedMemBytes  Amount of dynamic shared memory to allocate for this kernel.
# HIP-Clang compiler provides support for extern shared declarations.
# @param [in] stream  Stream where the kernel should be dispatched.
# @param [in] startEvent  If non-null, specified event will be updated to track the start time of
# the kernel launch. The event must be created before calling this API.
# @param [in] stopEvent  If non-null, specified event will be updated to track the stop time of
# the kernel launch. The event must be created before calling this API.
# May be 0, in which case the default stream is used with associated synchronization rules.
# @param [in] flags. The value of hipExtAnyOrderLaunch, signifies if kernel can be
# launched in any order.
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue.
cdef hipError_t hipExtLaunchKernel(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,int sharedMemBytes,hipStream_t stream,hipEvent_t startEvent,hipEvent_t stopEvent,int flags) nogil:
    global _lib_handle
    global hipExtLaunchKernel_funptr
    if hipExtLaunchKernel_funptr == NULL:
        with gil:
            hipExtLaunchKernel_funptr = loader.load_symbol(_lib_handle, "hipExtLaunchKernel")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,int,hipStream_t,hipEvent_t,hipEvent_t,int) nogil> hipExtLaunchKernel_funptr)(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream,startEvent,stopEvent,flags)


cdef void* hipBindTextureToMipmappedArray_funptr = NULL
# @brief  Binds a mipmapped array to a texture.
# @param [in] tex  pointer to the texture reference to bind
# @param [in] mipmappedArray  memory mipmapped array on the device
# @param [in] desc  opointer to the channel format
# @returns hipSuccess, hipErrorInvalidValue
cdef hipError_t hipBindTextureToMipmappedArray(textureReference * tex,hipMipmappedArray_const_t mipmappedArray,hipChannelFormatDesc * desc) nogil:
    global _lib_handle
    global hipBindTextureToMipmappedArray_funptr
    if hipBindTextureToMipmappedArray_funptr == NULL:
        with gil:
            hipBindTextureToMipmappedArray_funptr = loader.load_symbol(_lib_handle, "hipBindTextureToMipmappedArray")
    return (<hipError_t (*)(textureReference *,hipMipmappedArray_const_t,hipChannelFormatDesc *) nogil> hipBindTextureToMipmappedArray_funptr)(tex,mipmappedArray,desc)


cdef void* hipCreateTextureObject_funptr = NULL
# @brief Creates a texture object.
# @param [out] pTexObject  pointer to the texture object to create
# @param [in] pResDesc  pointer to resource descriptor
# @param [in] pTexDesc  pointer to texture descriptor
# @param [in] pResViewDesc  pointer to resource view descriptor
# @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported, hipErrorOutOfMemory
# @note 3D liner filter isn't supported on GFX90A boards, on which the API @p hipCreateTextureObject will
# return hipErrorNotSupported.
cdef hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject,hipResourceDesc * pResDesc,hipTextureDesc * pTexDesc,hipResourceViewDesc * pResViewDesc) nogil:
    global _lib_handle
    global hipCreateTextureObject_funptr
    if hipCreateTextureObject_funptr == NULL:
        with gil:
            hipCreateTextureObject_funptr = loader.load_symbol(_lib_handle, "hipCreateTextureObject")
    return (<hipError_t (*)(hipTextureObject_t*,hipResourceDesc *,hipTextureDesc *,hipResourceViewDesc *) nogil> hipCreateTextureObject_funptr)(pTexObject,pResDesc,pTexDesc,pResViewDesc)


cdef void* hipDestroyTextureObject_funptr = NULL
# @brief Destroys a texture object.
# @param [in] textureObject  texture object to destroy
# @returns hipSuccess, hipErrorInvalidValue
cdef hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) nogil:
    global _lib_handle
    global hipDestroyTextureObject_funptr
    if hipDestroyTextureObject_funptr == NULL:
        with gil:
            hipDestroyTextureObject_funptr = loader.load_symbol(_lib_handle, "hipDestroyTextureObject")
    return (<hipError_t (*)(hipTextureObject_t) nogil> hipDestroyTextureObject_funptr)(textureObject)


cdef void* hipGetChannelDesc_funptr = NULL
# @brief Gets the channel descriptor in an array.
# @param [in] desc  pointer to channel format descriptor
# @param [out] array  memory array on the device
# @returns hipSuccess, hipErrorInvalidValue
cdef hipError_t hipGetChannelDesc(hipChannelFormatDesc * desc,hipArray_const_t array) nogil:
    global _lib_handle
    global hipGetChannelDesc_funptr
    if hipGetChannelDesc_funptr == NULL:
        with gil:
            hipGetChannelDesc_funptr = loader.load_symbol(_lib_handle, "hipGetChannelDesc")
    return (<hipError_t (*)(hipChannelFormatDesc *,hipArray_const_t) nogil> hipGetChannelDesc_funptr)(desc,array)


cdef void* hipGetTextureObjectResourceDesc_funptr = NULL
# @brief Gets resource descriptor for the texture object.
# @param [out] pResDesc  pointer to resource descriptor
# @param [in] textureObject  texture object
# @returns hipSuccess, hipErrorInvalidValue
cdef hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc * pResDesc,hipTextureObject_t textureObject) nogil:
    global _lib_handle
    global hipGetTextureObjectResourceDesc_funptr
    if hipGetTextureObjectResourceDesc_funptr == NULL:
        with gil:
            hipGetTextureObjectResourceDesc_funptr = loader.load_symbol(_lib_handle, "hipGetTextureObjectResourceDesc")
    return (<hipError_t (*)(hipResourceDesc *,hipTextureObject_t) nogil> hipGetTextureObjectResourceDesc_funptr)(pResDesc,textureObject)


cdef void* hipGetTextureObjectResourceViewDesc_funptr = NULL
# @brief Gets resource view descriptor for the texture object.
# @param [out] pResViewDesc  pointer to resource view descriptor
# @param [in] textureObject  texture object
# @returns hipSuccess, hipErrorInvalidValue
cdef hipError_t hipGetTextureObjectResourceViewDesc(hipResourceViewDesc * pResViewDesc,hipTextureObject_t textureObject) nogil:
    global _lib_handle
    global hipGetTextureObjectResourceViewDesc_funptr
    if hipGetTextureObjectResourceViewDesc_funptr == NULL:
        with gil:
            hipGetTextureObjectResourceViewDesc_funptr = loader.load_symbol(_lib_handle, "hipGetTextureObjectResourceViewDesc")
    return (<hipError_t (*)(hipResourceViewDesc *,hipTextureObject_t) nogil> hipGetTextureObjectResourceViewDesc_funptr)(pResViewDesc,textureObject)


cdef void* hipGetTextureObjectTextureDesc_funptr = NULL
# @brief Gets texture descriptor for the texture object.
# @param [out] pTexDesc  pointer to texture descriptor
# @param [in] textureObject  texture object
# @returns hipSuccess, hipErrorInvalidValue
cdef hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc * pTexDesc,hipTextureObject_t textureObject) nogil:
    global _lib_handle
    global hipGetTextureObjectTextureDesc_funptr
    if hipGetTextureObjectTextureDesc_funptr == NULL:
        with gil:
            hipGetTextureObjectTextureDesc_funptr = loader.load_symbol(_lib_handle, "hipGetTextureObjectTextureDesc")
    return (<hipError_t (*)(hipTextureDesc *,hipTextureObject_t) nogil> hipGetTextureObjectTextureDesc_funptr)(pTexDesc,textureObject)


cdef void* hipTexObjectCreate_funptr = NULL
# @brief Creates a texture object.
# @param [out] pTexObject  pointer to texture object to create
# @param [in] pResDesc  pointer to resource descriptor
# @param [in] pTexDesc  pointer to texture descriptor
# @param [in] pResViewDesc  pointer to resource view descriptor
# @returns hipSuccess, hipErrorInvalidValue
cdef hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject,HIP_RESOURCE_DESC_st * pResDesc,HIP_TEXTURE_DESC_st * pTexDesc,HIP_RESOURCE_VIEW_DESC_st * pResViewDesc) nogil:
    global _lib_handle
    global hipTexObjectCreate_funptr
    if hipTexObjectCreate_funptr == NULL:
        with gil:
            hipTexObjectCreate_funptr = loader.load_symbol(_lib_handle, "hipTexObjectCreate")
    return (<hipError_t (*)(hipTextureObject_t*,HIP_RESOURCE_DESC_st *,HIP_TEXTURE_DESC_st *,HIP_RESOURCE_VIEW_DESC_st *) nogil> hipTexObjectCreate_funptr)(pTexObject,pResDesc,pTexDesc,pResViewDesc)


cdef void* hipTexObjectDestroy_funptr = NULL
# @brief Destroys a texture object.
# @param [in] texObject  texture object to destroy
# @returns hipSuccess, hipErrorInvalidValue
cdef hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) nogil:
    global _lib_handle
    global hipTexObjectDestroy_funptr
    if hipTexObjectDestroy_funptr == NULL:
        with gil:
            hipTexObjectDestroy_funptr = loader.load_symbol(_lib_handle, "hipTexObjectDestroy")
    return (<hipError_t (*)(hipTextureObject_t) nogil> hipTexObjectDestroy_funptr)(texObject)


cdef void* hipTexObjectGetResourceDesc_funptr = NULL
# @brief Gets resource descriptor of a texture object.
# @param [out] pResDesc  pointer to resource descriptor
# @param [in] texObject  texture object
# @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
cdef hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC_st * pResDesc,hipTextureObject_t texObject) nogil:
    global _lib_handle
    global hipTexObjectGetResourceDesc_funptr
    if hipTexObjectGetResourceDesc_funptr == NULL:
        with gil:
            hipTexObjectGetResourceDesc_funptr = loader.load_symbol(_lib_handle, "hipTexObjectGetResourceDesc")
    return (<hipError_t (*)(HIP_RESOURCE_DESC_st *,hipTextureObject_t) nogil> hipTexObjectGetResourceDesc_funptr)(pResDesc,texObject)


cdef void* hipTexObjectGetResourceViewDesc_funptr = NULL
# @brief Gets resource view descriptor of a texture object.
# @param [out] pResViewDesc  pointer to resource view descriptor
# @param [in] texObject  texture object
# @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
cdef hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC_st * pResViewDesc,hipTextureObject_t texObject) nogil:
    global _lib_handle
    global hipTexObjectGetResourceViewDesc_funptr
    if hipTexObjectGetResourceViewDesc_funptr == NULL:
        with gil:
            hipTexObjectGetResourceViewDesc_funptr = loader.load_symbol(_lib_handle, "hipTexObjectGetResourceViewDesc")
    return (<hipError_t (*)(HIP_RESOURCE_VIEW_DESC_st *,hipTextureObject_t) nogil> hipTexObjectGetResourceViewDesc_funptr)(pResViewDesc,texObject)


cdef void* hipTexObjectGetTextureDesc_funptr = NULL
# @brief Gets texture descriptor of a texture object.
# @param [out] pTexDesc  pointer to texture descriptor
# @param [in] texObject  texture object
# @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
cdef hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC_st * pTexDesc,hipTextureObject_t texObject) nogil:
    global _lib_handle
    global hipTexObjectGetTextureDesc_funptr
    if hipTexObjectGetTextureDesc_funptr == NULL:
        with gil:
            hipTexObjectGetTextureDesc_funptr = loader.load_symbol(_lib_handle, "hipTexObjectGetTextureDesc")
    return (<hipError_t (*)(HIP_TEXTURE_DESC_st *,hipTextureObject_t) nogil> hipTexObjectGetTextureDesc_funptr)(pTexDesc,texObject)


cdef void* hipGetTextureReference_funptr = NULL
# @addtogroup TextureD Texture Management [Deprecated]
# @{
# @ingroup Texture
# This section describes the deprecated texture management functions of HIP runtime API.
# @brief Gets the texture reference related with the symbol.
# @param [out] texref  texture reference
# @param [in] symbol  pointer to the symbol related with the texture for the reference
# @returns hipSuccess, hipErrorInvalidValue
cdef hipError_t hipGetTextureReference(textureReference ** texref,const void * symbol) nogil:
    global _lib_handle
    global hipGetTextureReference_funptr
    if hipGetTextureReference_funptr == NULL:
        with gil:
            hipGetTextureReference_funptr = loader.load_symbol(_lib_handle, "hipGetTextureReference")
    return (<hipError_t (*)(textureReference **,const void *) nogil> hipGetTextureReference_funptr)(texref,symbol)


cdef void* hipTexRefSetAddressMode_funptr = NULL
cdef hipError_t hipTexRefSetAddressMode(textureReference * texRef,int dim,hipTextureAddressMode am) nogil:
    global _lib_handle
    global hipTexRefSetAddressMode_funptr
    if hipTexRefSetAddressMode_funptr == NULL:
        with gil:
            hipTexRefSetAddressMode_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetAddressMode")
    return (<hipError_t (*)(textureReference *,int,hipTextureAddressMode) nogil> hipTexRefSetAddressMode_funptr)(texRef,dim,am)


cdef void* hipTexRefSetArray_funptr = NULL
cdef hipError_t hipTexRefSetArray(textureReference * tex,hipArray_const_t array,unsigned int flags) nogil:
    global _lib_handle
    global hipTexRefSetArray_funptr
    if hipTexRefSetArray_funptr == NULL:
        with gil:
            hipTexRefSetArray_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetArray")
    return (<hipError_t (*)(textureReference *,hipArray_const_t,unsigned int) nogil> hipTexRefSetArray_funptr)(tex,array,flags)


cdef void* hipTexRefSetFilterMode_funptr = NULL
cdef hipError_t hipTexRefSetFilterMode(textureReference * texRef,hipTextureFilterMode fm) nogil:
    global _lib_handle
    global hipTexRefSetFilterMode_funptr
    if hipTexRefSetFilterMode_funptr == NULL:
        with gil:
            hipTexRefSetFilterMode_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetFilterMode")
    return (<hipError_t (*)(textureReference *,hipTextureFilterMode) nogil> hipTexRefSetFilterMode_funptr)(texRef,fm)


cdef void* hipTexRefSetFlags_funptr = NULL
cdef hipError_t hipTexRefSetFlags(textureReference * texRef,unsigned int Flags) nogil:
    global _lib_handle
    global hipTexRefSetFlags_funptr
    if hipTexRefSetFlags_funptr == NULL:
        with gil:
            hipTexRefSetFlags_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetFlags")
    return (<hipError_t (*)(textureReference *,unsigned int) nogil> hipTexRefSetFlags_funptr)(texRef,Flags)


cdef void* hipTexRefSetFormat_funptr = NULL
cdef hipError_t hipTexRefSetFormat(textureReference * texRef,hipArray_Format fmt,int NumPackedComponents) nogil:
    global _lib_handle
    global hipTexRefSetFormat_funptr
    if hipTexRefSetFormat_funptr == NULL:
        with gil:
            hipTexRefSetFormat_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetFormat")
    return (<hipError_t (*)(textureReference *,hipArray_Format,int) nogil> hipTexRefSetFormat_funptr)(texRef,fmt,NumPackedComponents)


cdef void* hipBindTexture_funptr = NULL
cdef hipError_t hipBindTexture(int * offset,textureReference * tex,const void * devPtr,hipChannelFormatDesc * desc,int size) nogil:
    global _lib_handle
    global hipBindTexture_funptr
    if hipBindTexture_funptr == NULL:
        with gil:
            hipBindTexture_funptr = loader.load_symbol(_lib_handle, "hipBindTexture")
    return (<hipError_t (*)(int *,textureReference *,const void *,hipChannelFormatDesc *,int) nogil> hipBindTexture_funptr)(offset,tex,devPtr,desc,size)


cdef void* hipBindTexture2D_funptr = NULL
cdef hipError_t hipBindTexture2D(int * offset,textureReference * tex,const void * devPtr,hipChannelFormatDesc * desc,int width,int height,int pitch) nogil:
    global _lib_handle
    global hipBindTexture2D_funptr
    if hipBindTexture2D_funptr == NULL:
        with gil:
            hipBindTexture2D_funptr = loader.load_symbol(_lib_handle, "hipBindTexture2D")
    return (<hipError_t (*)(int *,textureReference *,const void *,hipChannelFormatDesc *,int,int,int) nogil> hipBindTexture2D_funptr)(offset,tex,devPtr,desc,width,height,pitch)


cdef void* hipBindTextureToArray_funptr = NULL
cdef hipError_t hipBindTextureToArray(textureReference * tex,hipArray_const_t array,hipChannelFormatDesc * desc) nogil:
    global _lib_handle
    global hipBindTextureToArray_funptr
    if hipBindTextureToArray_funptr == NULL:
        with gil:
            hipBindTextureToArray_funptr = loader.load_symbol(_lib_handle, "hipBindTextureToArray")
    return (<hipError_t (*)(textureReference *,hipArray_const_t,hipChannelFormatDesc *) nogil> hipBindTextureToArray_funptr)(tex,array,desc)


cdef void* hipGetTextureAlignmentOffset_funptr = NULL
cdef hipError_t hipGetTextureAlignmentOffset(int * offset,textureReference * texref) nogil:
    global _lib_handle
    global hipGetTextureAlignmentOffset_funptr
    if hipGetTextureAlignmentOffset_funptr == NULL:
        with gil:
            hipGetTextureAlignmentOffset_funptr = loader.load_symbol(_lib_handle, "hipGetTextureAlignmentOffset")
    return (<hipError_t (*)(int *,textureReference *) nogil> hipGetTextureAlignmentOffset_funptr)(offset,texref)


cdef void* hipUnbindTexture_funptr = NULL
cdef hipError_t hipUnbindTexture(textureReference * tex) nogil:
    global _lib_handle
    global hipUnbindTexture_funptr
    if hipUnbindTexture_funptr == NULL:
        with gil:
            hipUnbindTexture_funptr = loader.load_symbol(_lib_handle, "hipUnbindTexture")
    return (<hipError_t (*)(textureReference *) nogil> hipUnbindTexture_funptr)(tex)


cdef void* hipTexRefGetAddress_funptr = NULL
cdef hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr,textureReference * texRef) nogil:
    global _lib_handle
    global hipTexRefGetAddress_funptr
    if hipTexRefGetAddress_funptr == NULL:
        with gil:
            hipTexRefGetAddress_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetAddress")
    return (<hipError_t (*)(hipDeviceptr_t*,textureReference *) nogil> hipTexRefGetAddress_funptr)(dev_ptr,texRef)


cdef void* hipTexRefGetAddressMode_funptr = NULL
cdef hipError_t hipTexRefGetAddressMode(hipTextureAddressMode * pam,textureReference * texRef,int dim) nogil:
    global _lib_handle
    global hipTexRefGetAddressMode_funptr
    if hipTexRefGetAddressMode_funptr == NULL:
        with gil:
            hipTexRefGetAddressMode_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetAddressMode")
    return (<hipError_t (*)(hipTextureAddressMode *,textureReference *,int) nogil> hipTexRefGetAddressMode_funptr)(pam,texRef,dim)


cdef void* hipTexRefGetFilterMode_funptr = NULL
cdef hipError_t hipTexRefGetFilterMode(hipTextureFilterMode * pfm,textureReference * texRef) nogil:
    global _lib_handle
    global hipTexRefGetFilterMode_funptr
    if hipTexRefGetFilterMode_funptr == NULL:
        with gil:
            hipTexRefGetFilterMode_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetFilterMode")
    return (<hipError_t (*)(hipTextureFilterMode *,textureReference *) nogil> hipTexRefGetFilterMode_funptr)(pfm,texRef)


cdef void* hipTexRefGetFlags_funptr = NULL
cdef hipError_t hipTexRefGetFlags(unsigned int * pFlags,textureReference * texRef) nogil:
    global _lib_handle
    global hipTexRefGetFlags_funptr
    if hipTexRefGetFlags_funptr == NULL:
        with gil:
            hipTexRefGetFlags_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetFlags")
    return (<hipError_t (*)(unsigned int *,textureReference *) nogil> hipTexRefGetFlags_funptr)(pFlags,texRef)


cdef void* hipTexRefGetFormat_funptr = NULL
cdef hipError_t hipTexRefGetFormat(hipArray_Format * pFormat,int * pNumChannels,textureReference * texRef) nogil:
    global _lib_handle
    global hipTexRefGetFormat_funptr
    if hipTexRefGetFormat_funptr == NULL:
        with gil:
            hipTexRefGetFormat_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetFormat")
    return (<hipError_t (*)(hipArray_Format *,int *,textureReference *) nogil> hipTexRefGetFormat_funptr)(pFormat,pNumChannels,texRef)


cdef void* hipTexRefGetMaxAnisotropy_funptr = NULL
cdef hipError_t hipTexRefGetMaxAnisotropy(int * pmaxAnsio,textureReference * texRef) nogil:
    global _lib_handle
    global hipTexRefGetMaxAnisotropy_funptr
    if hipTexRefGetMaxAnisotropy_funptr == NULL:
        with gil:
            hipTexRefGetMaxAnisotropy_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetMaxAnisotropy")
    return (<hipError_t (*)(int *,textureReference *) nogil> hipTexRefGetMaxAnisotropy_funptr)(pmaxAnsio,texRef)


cdef void* hipTexRefGetMipmapFilterMode_funptr = NULL
cdef hipError_t hipTexRefGetMipmapFilterMode(hipTextureFilterMode * pfm,textureReference * texRef) nogil:
    global _lib_handle
    global hipTexRefGetMipmapFilterMode_funptr
    if hipTexRefGetMipmapFilterMode_funptr == NULL:
        with gil:
            hipTexRefGetMipmapFilterMode_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetMipmapFilterMode")
    return (<hipError_t (*)(hipTextureFilterMode *,textureReference *) nogil> hipTexRefGetMipmapFilterMode_funptr)(pfm,texRef)


cdef void* hipTexRefGetMipmapLevelBias_funptr = NULL
cdef hipError_t hipTexRefGetMipmapLevelBias(float * pbias,textureReference * texRef) nogil:
    global _lib_handle
    global hipTexRefGetMipmapLevelBias_funptr
    if hipTexRefGetMipmapLevelBias_funptr == NULL:
        with gil:
            hipTexRefGetMipmapLevelBias_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetMipmapLevelBias")
    return (<hipError_t (*)(float *,textureReference *) nogil> hipTexRefGetMipmapLevelBias_funptr)(pbias,texRef)


cdef void* hipTexRefGetMipmapLevelClamp_funptr = NULL
cdef hipError_t hipTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp,float * pmaxMipmapLevelClamp,textureReference * texRef) nogil:
    global _lib_handle
    global hipTexRefGetMipmapLevelClamp_funptr
    if hipTexRefGetMipmapLevelClamp_funptr == NULL:
        with gil:
            hipTexRefGetMipmapLevelClamp_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetMipmapLevelClamp")
    return (<hipError_t (*)(float *,float *,textureReference *) nogil> hipTexRefGetMipmapLevelClamp_funptr)(pminMipmapLevelClamp,pmaxMipmapLevelClamp,texRef)


cdef void* hipTexRefGetMipMappedArray_funptr = NULL
cdef hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray,textureReference * texRef) nogil:
    global _lib_handle
    global hipTexRefGetMipMappedArray_funptr
    if hipTexRefGetMipMappedArray_funptr == NULL:
        with gil:
            hipTexRefGetMipMappedArray_funptr = loader.load_symbol(_lib_handle, "hipTexRefGetMipMappedArray")
    return (<hipError_t (*)(hipMipmappedArray_t*,textureReference *) nogil> hipTexRefGetMipMappedArray_funptr)(pArray,texRef)


cdef void* hipTexRefSetAddress_funptr = NULL
cdef hipError_t hipTexRefSetAddress(int * ByteOffset,textureReference * texRef,hipDeviceptr_t dptr,int bytes) nogil:
    global _lib_handle
    global hipTexRefSetAddress_funptr
    if hipTexRefSetAddress_funptr == NULL:
        with gil:
            hipTexRefSetAddress_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetAddress")
    return (<hipError_t (*)(int *,textureReference *,hipDeviceptr_t,int) nogil> hipTexRefSetAddress_funptr)(ByteOffset,texRef,dptr,bytes)


cdef void* hipTexRefSetAddress2D_funptr = NULL
cdef hipError_t hipTexRefSetAddress2D(textureReference * texRef,HIP_ARRAY_DESCRIPTOR * desc,hipDeviceptr_t dptr,int Pitch) nogil:
    global _lib_handle
    global hipTexRefSetAddress2D_funptr
    if hipTexRefSetAddress2D_funptr == NULL:
        with gil:
            hipTexRefSetAddress2D_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetAddress2D")
    return (<hipError_t (*)(textureReference *,HIP_ARRAY_DESCRIPTOR *,hipDeviceptr_t,int) nogil> hipTexRefSetAddress2D_funptr)(texRef,desc,dptr,Pitch)


cdef void* hipTexRefSetMaxAnisotropy_funptr = NULL
cdef hipError_t hipTexRefSetMaxAnisotropy(textureReference * texRef,unsigned int maxAniso) nogil:
    global _lib_handle
    global hipTexRefSetMaxAnisotropy_funptr
    if hipTexRefSetMaxAnisotropy_funptr == NULL:
        with gil:
            hipTexRefSetMaxAnisotropy_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetMaxAnisotropy")
    return (<hipError_t (*)(textureReference *,unsigned int) nogil> hipTexRefSetMaxAnisotropy_funptr)(texRef,maxAniso)


cdef void* hipTexRefSetBorderColor_funptr = NULL
cdef hipError_t hipTexRefSetBorderColor(textureReference * texRef,float * pBorderColor) nogil:
    global _lib_handle
    global hipTexRefSetBorderColor_funptr
    if hipTexRefSetBorderColor_funptr == NULL:
        with gil:
            hipTexRefSetBorderColor_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetBorderColor")
    return (<hipError_t (*)(textureReference *,float *) nogil> hipTexRefSetBorderColor_funptr)(texRef,pBorderColor)


cdef void* hipTexRefSetMipmapFilterMode_funptr = NULL
cdef hipError_t hipTexRefSetMipmapFilterMode(textureReference * texRef,hipTextureFilterMode fm) nogil:
    global _lib_handle
    global hipTexRefSetMipmapFilterMode_funptr
    if hipTexRefSetMipmapFilterMode_funptr == NULL:
        with gil:
            hipTexRefSetMipmapFilterMode_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetMipmapFilterMode")
    return (<hipError_t (*)(textureReference *,hipTextureFilterMode) nogil> hipTexRefSetMipmapFilterMode_funptr)(texRef,fm)


cdef void* hipTexRefSetMipmapLevelBias_funptr = NULL
cdef hipError_t hipTexRefSetMipmapLevelBias(textureReference * texRef,float bias) nogil:
    global _lib_handle
    global hipTexRefSetMipmapLevelBias_funptr
    if hipTexRefSetMipmapLevelBias_funptr == NULL:
        with gil:
            hipTexRefSetMipmapLevelBias_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetMipmapLevelBias")
    return (<hipError_t (*)(textureReference *,float) nogil> hipTexRefSetMipmapLevelBias_funptr)(texRef,bias)


cdef void* hipTexRefSetMipmapLevelClamp_funptr = NULL
cdef hipError_t hipTexRefSetMipmapLevelClamp(textureReference * texRef,float minMipMapLevelClamp,float maxMipMapLevelClamp) nogil:
    global _lib_handle
    global hipTexRefSetMipmapLevelClamp_funptr
    if hipTexRefSetMipmapLevelClamp_funptr == NULL:
        with gil:
            hipTexRefSetMipmapLevelClamp_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetMipmapLevelClamp")
    return (<hipError_t (*)(textureReference *,float,float) nogil> hipTexRefSetMipmapLevelClamp_funptr)(texRef,minMipMapLevelClamp,maxMipMapLevelClamp)


cdef void* hipTexRefSetMipmappedArray_funptr = NULL
cdef hipError_t hipTexRefSetMipmappedArray(textureReference * texRef,hipMipmappedArray * mipmappedArray,unsigned int Flags) nogil:
    global _lib_handle
    global hipTexRefSetMipmappedArray_funptr
    if hipTexRefSetMipmappedArray_funptr == NULL:
        with gil:
            hipTexRefSetMipmappedArray_funptr = loader.load_symbol(_lib_handle, "hipTexRefSetMipmappedArray")
    return (<hipError_t (*)(textureReference *,hipMipmappedArray *,unsigned int) nogil> hipTexRefSetMipmappedArray_funptr)(texRef,mipmappedArray,Flags)


cdef void* hipMipmappedArrayCreate_funptr = NULL
# @addtogroup TextureU Texture Management [Not supported]
# @{
# @ingroup Texture
# This section describes the texture management functions currently unsupported in HIP runtime.
cdef hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,HIP_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc,unsigned int numMipmapLevels) nogil:
    global _lib_handle
    global hipMipmappedArrayCreate_funptr
    if hipMipmappedArrayCreate_funptr == NULL:
        with gil:
            hipMipmappedArrayCreate_funptr = loader.load_symbol(_lib_handle, "hipMipmappedArrayCreate")
    return (<hipError_t (*)(hipMipmappedArray_t*,HIP_ARRAY3D_DESCRIPTOR *,unsigned int) nogil> hipMipmappedArrayCreate_funptr)(pHandle,pMipmappedArrayDesc,numMipmapLevels)


cdef void* hipMipmappedArrayDestroy_funptr = NULL
cdef hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) nogil:
    global _lib_handle
    global hipMipmappedArrayDestroy_funptr
    if hipMipmappedArrayDestroy_funptr == NULL:
        with gil:
            hipMipmappedArrayDestroy_funptr = loader.load_symbol(_lib_handle, "hipMipmappedArrayDestroy")
    return (<hipError_t (*)(hipMipmappedArray_t) nogil> hipMipmappedArrayDestroy_funptr)(hMipmappedArray)


cdef void* hipMipmappedArrayGetLevel_funptr = NULL
cdef hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray,hipMipmappedArray_t hMipMappedArray,unsigned int level) nogil:
    global _lib_handle
    global hipMipmappedArrayGetLevel_funptr
    if hipMipmappedArrayGetLevel_funptr == NULL:
        with gil:
            hipMipmappedArrayGetLevel_funptr = loader.load_symbol(_lib_handle, "hipMipmappedArrayGetLevel")
    return (<hipError_t (*)(hipArray_t*,hipMipmappedArray_t,unsigned int) nogil> hipMipmappedArrayGetLevel_funptr)(pLevelArray,hMipMappedArray,level)


cdef void* hipApiName_funptr = NULL
# @defgroup Callback Callback Activity APIs
# @{
# This section describes the callback/Activity of HIP runtime API.
cdef const char * hipApiName(uint32_t id) nogil:
    global _lib_handle
    global hipApiName_funptr
    if hipApiName_funptr == NULL:
        with gil:
            hipApiName_funptr = loader.load_symbol(_lib_handle, "hipApiName")
    return (<const char * (*)(uint32_t) nogil> hipApiName_funptr)(id)


cdef void* hipKernelNameRef_funptr = NULL
cdef const char * hipKernelNameRef(hipFunction_t f) nogil:
    global _lib_handle
    global hipKernelNameRef_funptr
    if hipKernelNameRef_funptr == NULL:
        with gil:
            hipKernelNameRef_funptr = loader.load_symbol(_lib_handle, "hipKernelNameRef")
    return (<const char * (*)(hipFunction_t) nogil> hipKernelNameRef_funptr)(f)


cdef void* hipKernelNameRefByPtr_funptr = NULL
cdef const char * hipKernelNameRefByPtr(const void * hostFunction,hipStream_t stream) nogil:
    global _lib_handle
    global hipKernelNameRefByPtr_funptr
    if hipKernelNameRefByPtr_funptr == NULL:
        with gil:
            hipKernelNameRefByPtr_funptr = loader.load_symbol(_lib_handle, "hipKernelNameRefByPtr")
    return (<const char * (*)(const void *,hipStream_t) nogil> hipKernelNameRefByPtr_funptr)(hostFunction,stream)


cdef void* hipGetStreamDeviceId_funptr = NULL
cdef int hipGetStreamDeviceId(hipStream_t stream) nogil:
    global _lib_handle
    global hipGetStreamDeviceId_funptr
    if hipGetStreamDeviceId_funptr == NULL:
        with gil:
            hipGetStreamDeviceId_funptr = loader.load_symbol(_lib_handle, "hipGetStreamDeviceId")
    return (<int (*)(hipStream_t) nogil> hipGetStreamDeviceId_funptr)(stream)


cdef void* hipStreamBeginCapture_funptr = NULL
# @brief Begins graph capture on a stream.
# @param [in] stream - Stream to initiate capture.
# @param [in] mode - Controls the interaction of this capture sequence with other API calls that
# are not safe.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipStreamBeginCapture(hipStream_t stream,hipStreamCaptureMode mode) nogil:
    global _lib_handle
    global hipStreamBeginCapture_funptr
    if hipStreamBeginCapture_funptr == NULL:
        with gil:
            hipStreamBeginCapture_funptr = loader.load_symbol(_lib_handle, "hipStreamBeginCapture")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureMode) nogil> hipStreamBeginCapture_funptr)(stream,mode)


cdef void* hipStreamEndCapture_funptr = NULL
# @brief Ends capture on a stream, returning the captured graph.
# @param [in] stream - Stream to end capture.
# @param [out] pGraph - returns the graph captured.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipStreamEndCapture(hipStream_t stream,hipGraph_t* pGraph) nogil:
    global _lib_handle
    global hipStreamEndCapture_funptr
    if hipStreamEndCapture_funptr == NULL:
        with gil:
            hipStreamEndCapture_funptr = loader.load_symbol(_lib_handle, "hipStreamEndCapture")
    return (<hipError_t (*)(hipStream_t,hipGraph_t*) nogil> hipStreamEndCapture_funptr)(stream,pGraph)


cdef void* hipStreamGetCaptureInfo_funptr = NULL
# @brief Get capture status of a stream.
# @param [in] stream - Stream under capture.
# @param [out] pCaptureStatus - returns current status of the capture.
# @param [out] pId - unique ID of the capture.
# @returns #hipSuccess, #hipErrorStreamCaptureImplicit
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipStreamGetCaptureInfo(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus,unsigned long long * pId) nogil:
    global _lib_handle
    global hipStreamGetCaptureInfo_funptr
    if hipStreamGetCaptureInfo_funptr == NULL:
        with gil:
            hipStreamGetCaptureInfo_funptr = loader.load_symbol(_lib_handle, "hipStreamGetCaptureInfo")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *) nogil> hipStreamGetCaptureInfo_funptr)(stream,pCaptureStatus,pId)


cdef void* hipStreamGetCaptureInfo_v2_funptr = NULL
# @brief Get stream's capture state
# @param [in] stream - Stream under capture.
# @param [out] captureStatus_out - returns current status of the capture.
# @param [out] id_out - unique ID of the capture.
# @param [in] graph_out - returns the graph being captured into.
# @param [out] dependencies_out - returns pointer to an array of nodes.
# @param [out] numDependencies_out - returns size of the array returned in dependencies_out.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream,hipStreamCaptureStatus * captureStatus_out,unsigned long long * id_out,hipGraph_t* graph_out,hipGraphNode_t ** dependencies_out,int * numDependencies_out) nogil:
    global _lib_handle
    global hipStreamGetCaptureInfo_v2_funptr
    if hipStreamGetCaptureInfo_v2_funptr == NULL:
        with gil:
            hipStreamGetCaptureInfo_v2_funptr = loader.load_symbol(_lib_handle, "hipStreamGetCaptureInfo_v2")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *,hipGraph_t*,hipGraphNode_t **,int *) nogil> hipStreamGetCaptureInfo_v2_funptr)(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out)


cdef void* hipStreamIsCapturing_funptr = NULL
# @brief Get stream's capture state
# @param [in] stream - Stream under capture.
# @param [out] pCaptureStatus - returns current status of the capture.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipStreamIsCapturing(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus) nogil:
    global _lib_handle
    global hipStreamIsCapturing_funptr
    if hipStreamIsCapturing_funptr == NULL:
        with gil:
            hipStreamIsCapturing_funptr = loader.load_symbol(_lib_handle, "hipStreamIsCapturing")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *) nogil> hipStreamIsCapturing_funptr)(stream,pCaptureStatus)


cdef void* hipStreamUpdateCaptureDependencies_funptr = NULL
# @brief Update the set of dependencies in a capturing stream
# @param [in] stream - Stream under capture.
# @param [in] dependencies - pointer to an array of nodes to Add/Replace.
# @param [in] numDependencies - size of the array in dependencies.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorIllegalState
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream,hipGraphNode_t* dependencies,int numDependencies,unsigned int flags) nogil:
    global _lib_handle
    global hipStreamUpdateCaptureDependencies_funptr
    if hipStreamUpdateCaptureDependencies_funptr == NULL:
        with gil:
            hipStreamUpdateCaptureDependencies_funptr = loader.load_symbol(_lib_handle, "hipStreamUpdateCaptureDependencies")
    return (<hipError_t (*)(hipStream_t,hipGraphNode_t*,int,unsigned int) nogil> hipStreamUpdateCaptureDependencies_funptr)(stream,dependencies,numDependencies,flags)


cdef void* hipThreadExchangeStreamCaptureMode_funptr = NULL
# @brief Swaps the stream capture mode of a thread.
# @param [in] mode - Pointer to mode value to swap with the current mode
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode * mode) nogil:
    global _lib_handle
    global hipThreadExchangeStreamCaptureMode_funptr
    if hipThreadExchangeStreamCaptureMode_funptr == NULL:
        with gil:
            hipThreadExchangeStreamCaptureMode_funptr = loader.load_symbol(_lib_handle, "hipThreadExchangeStreamCaptureMode")
    return (<hipError_t (*)(hipStreamCaptureMode *) nogil> hipThreadExchangeStreamCaptureMode_funptr)(mode)


cdef void* hipGraphCreate_funptr = NULL
# @brief Creates a graph
# @param [out] pGraph - pointer to graph to create.
# @param [in] flags - flags for graph creation, must be 0.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphCreate(hipGraph_t* pGraph,unsigned int flags) nogil:
    global _lib_handle
    global hipGraphCreate_funptr
    if hipGraphCreate_funptr == NULL:
        with gil:
            hipGraphCreate_funptr = loader.load_symbol(_lib_handle, "hipGraphCreate")
    return (<hipError_t (*)(hipGraph_t*,unsigned int) nogil> hipGraphCreate_funptr)(pGraph,flags)


cdef void* hipGraphDestroy_funptr = NULL
# @brief Destroys a graph
# @param [in] graph - instance of graph to destroy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphDestroy(hipGraph_t graph) nogil:
    global _lib_handle
    global hipGraphDestroy_funptr
    if hipGraphDestroy_funptr == NULL:
        with gil:
            hipGraphDestroy_funptr = loader.load_symbol(_lib_handle, "hipGraphDestroy")
    return (<hipError_t (*)(hipGraph_t) nogil> hipGraphDestroy_funptr)(graph)


cdef void* hipGraphAddDependencies_funptr = NULL
# @brief Adds dependency edges to a graph.
# @param [in] graph - instance of the graph to add dependencies.
# @param [in] from - pointer to the graph nodes with dependenties to add from.
# @param [in] to - pointer to the graph nodes to add dependenties to.
# @param [in] numDependencies - the number of dependencies to add.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddDependencies(hipGraph_t graph,hipGraphNode_t * from_,hipGraphNode_t * to,int numDependencies) nogil:
    global _lib_handle
    global hipGraphAddDependencies_funptr
    if hipGraphAddDependencies_funptr == NULL:
        with gil:
            hipGraphAddDependencies_funptr = loader.load_symbol(_lib_handle, "hipGraphAddDependencies")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t *,hipGraphNode_t *,int) nogil> hipGraphAddDependencies_funptr)(graph,from_,to,numDependencies)


cdef void* hipGraphRemoveDependencies_funptr = NULL
# @brief Removes dependency edges from a graph.
# @param [in] graph - instance of the graph to remove dependencies.
# @param [in] from - Array of nodes that provide the dependencies.
# @param [in] to - Array of dependent nodes.
# @param [in] numDependencies - the number of dependencies to remove.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphRemoveDependencies(hipGraph_t graph,hipGraphNode_t * from_,hipGraphNode_t * to,int numDependencies) nogil:
    global _lib_handle
    global hipGraphRemoveDependencies_funptr
    if hipGraphRemoveDependencies_funptr == NULL:
        with gil:
            hipGraphRemoveDependencies_funptr = loader.load_symbol(_lib_handle, "hipGraphRemoveDependencies")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t *,hipGraphNode_t *,int) nogil> hipGraphRemoveDependencies_funptr)(graph,from_,to,numDependencies)


cdef void* hipGraphGetEdges_funptr = NULL
# @brief Returns a graph's dependency edges.
# @param [in] graph - instance of the graph to get the edges from.
# @param [out] from - pointer to the graph nodes to return edge endpoints.
# @param [out] to - pointer to the graph nodes to return edge endpoints.
# @param [out] numEdges - returns number of edges.
# @returns #hipSuccess, #hipErrorInvalidValue
# from and to may both be NULL, in which case this function only returns the number of edges in
# numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher than the actual
# number of edges, the remaining entries in from and to will be set to NULL, and the number of
# edges actually returned will be written to numEdges
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphGetEdges(hipGraph_t graph,hipGraphNode_t* from_,hipGraphNode_t* to,int * numEdges) nogil:
    global _lib_handle
    global hipGraphGetEdges_funptr
    if hipGraphGetEdges_funptr == NULL:
        with gil:
            hipGraphGetEdges_funptr = loader.load_symbol(_lib_handle, "hipGraphGetEdges")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t*,hipGraphNode_t*,int *) nogil> hipGraphGetEdges_funptr)(graph,from_,to,numEdges)


cdef void* hipGraphGetNodes_funptr = NULL
# @brief Returns graph nodes.
# @param [in] graph - instance of graph to get the nodes.
# @param [out] nodes - pointer to return the  graph nodes.
# @param [out] numNodes - returns number of graph nodes.
# @returns #hipSuccess, #hipErrorInvalidValue
# nodes may be NULL, in which case this function will return the number of nodes in numNodes.
# Otherwise, numNodes entries will be filled in. If numNodes is higher than the actual number of
# nodes, the remaining entries in nodes will be set to NULL, and the number of nodes actually
# obtained will be returned in numNodes.
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphGetNodes(hipGraph_t graph,hipGraphNode_t* nodes,int * numNodes) nogil:
    global _lib_handle
    global hipGraphGetNodes_funptr
    if hipGraphGetNodes_funptr == NULL:
        with gil:
            hipGraphGetNodes_funptr = loader.load_symbol(_lib_handle, "hipGraphGetNodes")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t*,int *) nogil> hipGraphGetNodes_funptr)(graph,nodes,numNodes)


cdef void* hipGraphGetRootNodes_funptr = NULL
# @brief Returns graph's root nodes.
# @param [in] graph - instance of the graph to get the nodes.
# @param [out] pRootNodes - pointer to return the graph's root nodes.
# @param [out] pNumRootNodes - returns the number of graph's root nodes.
# @returns #hipSuccess, #hipErrorInvalidValue
# pRootNodes may be NULL, in which case this function will return the number of root nodes in
# pNumRootNodes. Otherwise, pNumRootNodes entries will be filled in. If pNumRootNodes is higher
# than the actual number of root nodes, the remaining entries in pRootNodes will be set to NULL,
# and the number of nodes actually obtained will be returned in pNumRootNodes.
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphGetRootNodes(hipGraph_t graph,hipGraphNode_t* pRootNodes,int * pNumRootNodes) nogil:
    global _lib_handle
    global hipGraphGetRootNodes_funptr
    if hipGraphGetRootNodes_funptr == NULL:
        with gil:
            hipGraphGetRootNodes_funptr = loader.load_symbol(_lib_handle, "hipGraphGetRootNodes")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t*,int *) nogil> hipGraphGetRootNodes_funptr)(graph,pRootNodes,pNumRootNodes)


cdef void* hipGraphNodeGetDependencies_funptr = NULL
# @brief Returns a node's dependencies.
# @param [in] node - graph node to get the dependencies from.
# @param [out] pDependencies - pointer to to return the dependencies.
# @param [out] pNumDependencies -  returns the number of graph node dependencies.
# @returns #hipSuccess, #hipErrorInvalidValue
# pDependencies may be NULL, in which case this function will return the number of dependencies in
# pNumDependencies. Otherwise, pNumDependencies entries will be filled in. If pNumDependencies is
# higher than the actual number of dependencies, the remaining entries in pDependencies will be set
# to NULL, and the number of nodes actually obtained will be returned in pNumDependencies.
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node,hipGraphNode_t* pDependencies,int * pNumDependencies) nogil:
    global _lib_handle
    global hipGraphNodeGetDependencies_funptr
    if hipGraphNodeGetDependencies_funptr == NULL:
        with gil:
            hipGraphNodeGetDependencies_funptr = loader.load_symbol(_lib_handle, "hipGraphNodeGetDependencies")
    return (<hipError_t (*)(hipGraphNode_t,hipGraphNode_t*,int *) nogil> hipGraphNodeGetDependencies_funptr)(node,pDependencies,pNumDependencies)


cdef void* hipGraphNodeGetDependentNodes_funptr = NULL
# @brief Returns a node's dependent nodes.
# @param [in] node - graph node to get the Dependent nodes from.
# @param [out] pDependentNodes - pointer to return the graph dependent nodes.
# @param [out] pNumDependentNodes - returns the number of graph node dependent nodes.
# @returns #hipSuccess, #hipErrorInvalidValue
# DependentNodes may be NULL, in which case this function will return the number of dependent nodes
# in pNumDependentNodes. Otherwise, pNumDependentNodes entries will be filled in. If
# pNumDependentNodes is higher than the actual number of dependent nodes, the remaining entries in
# pDependentNodes will be set to NULL, and the number of nodes actually obtained will be returned
# in pNumDependentNodes.
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node,hipGraphNode_t* pDependentNodes,int * pNumDependentNodes) nogil:
    global _lib_handle
    global hipGraphNodeGetDependentNodes_funptr
    if hipGraphNodeGetDependentNodes_funptr == NULL:
        with gil:
            hipGraphNodeGetDependentNodes_funptr = loader.load_symbol(_lib_handle, "hipGraphNodeGetDependentNodes")
    return (<hipError_t (*)(hipGraphNode_t,hipGraphNode_t*,int *) nogil> hipGraphNodeGetDependentNodes_funptr)(node,pDependentNodes,pNumDependentNodes)


cdef void* hipGraphNodeGetType_funptr = NULL
# @brief Returns a node's type.
# @param [in] node - instance of the graph to add dependencies.
# @param [out] pType - pointer to the return the type
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphNodeGetType(hipGraphNode_t node,hipGraphNodeType * pType) nogil:
    global _lib_handle
    global hipGraphNodeGetType_funptr
    if hipGraphNodeGetType_funptr == NULL:
        with gil:
            hipGraphNodeGetType_funptr = loader.load_symbol(_lib_handle, "hipGraphNodeGetType")
    return (<hipError_t (*)(hipGraphNode_t,hipGraphNodeType *) nogil> hipGraphNodeGetType_funptr)(node,pType)


cdef void* hipGraphDestroyNode_funptr = NULL
# @brief Remove a node from the graph.
# @param [in] node - graph node to remove
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphDestroyNode(hipGraphNode_t node) nogil:
    global _lib_handle
    global hipGraphDestroyNode_funptr
    if hipGraphDestroyNode_funptr == NULL:
        with gil:
            hipGraphDestroyNode_funptr = loader.load_symbol(_lib_handle, "hipGraphDestroyNode")
    return (<hipError_t (*)(hipGraphNode_t) nogil> hipGraphDestroyNode_funptr)(node)


cdef void* hipGraphClone_funptr = NULL
# @brief Clones a graph.
# @param [out] pGraphClone - Returns newly created cloned graph.
# @param [in] originalGraph - original graph to clone from.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphClone(hipGraph_t* pGraphClone,hipGraph_t originalGraph) nogil:
    global _lib_handle
    global hipGraphClone_funptr
    if hipGraphClone_funptr == NULL:
        with gil:
            hipGraphClone_funptr = loader.load_symbol(_lib_handle, "hipGraphClone")
    return (<hipError_t (*)(hipGraph_t*,hipGraph_t) nogil> hipGraphClone_funptr)(pGraphClone,originalGraph)


cdef void* hipGraphNodeFindInClone_funptr = NULL
# @brief Finds a cloned version of a node.
# @param [out] pNode - Returns the cloned node.
# @param [in] originalNode - original node handle.
# @param [in] clonedGraph - Cloned graph to query.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode,hipGraphNode_t originalNode,hipGraph_t clonedGraph) nogil:
    global _lib_handle
    global hipGraphNodeFindInClone_funptr
    if hipGraphNodeFindInClone_funptr == NULL:
        with gil:
            hipGraphNodeFindInClone_funptr = loader.load_symbol(_lib_handle, "hipGraphNodeFindInClone")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraphNode_t,hipGraph_t) nogil> hipGraphNodeFindInClone_funptr)(pNode,originalNode,clonedGraph)


cdef void* hipGraphInstantiate_funptr = NULL
# @brief Creates an executable graph from a graph
# @param [out] pGraphExec - pointer to instantiated executable graph that is created.
# @param [in] graph - instance of graph to instantiate.
# @param [out] pErrorNode - pointer to error node in case error occured in graph instantiation,
# it could modify the correponding node.
# @param [out] pLogBuffer - pointer to log buffer.
# @param [out] bufferSize - the size of log buffer.
# @returns #hipSuccess, #hipErrorOutOfMemory
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec,hipGraph_t graph,hipGraphNode_t* pErrorNode,char * pLogBuffer,int bufferSize) nogil:
    global _lib_handle
    global hipGraphInstantiate_funptr
    if hipGraphInstantiate_funptr == NULL:
        with gil:
            hipGraphInstantiate_funptr = loader.load_symbol(_lib_handle, "hipGraphInstantiate")
    return (<hipError_t (*)(hipGraphExec_t*,hipGraph_t,hipGraphNode_t*,char *,int) nogil> hipGraphInstantiate_funptr)(pGraphExec,graph,pErrorNode,pLogBuffer,bufferSize)


cdef void* hipGraphInstantiateWithFlags_funptr = NULL
# @brief Creates an executable graph from a graph.
# @param [out] pGraphExec - pointer to instantiated executable graph that is created.
# @param [in] graph - instance of graph to instantiate.
# @param [in] flags - Flags to control instantiation.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec,hipGraph_t graph,unsigned long long flags) nogil:
    global _lib_handle
    global hipGraphInstantiateWithFlags_funptr
    if hipGraphInstantiateWithFlags_funptr == NULL:
        with gil:
            hipGraphInstantiateWithFlags_funptr = loader.load_symbol(_lib_handle, "hipGraphInstantiateWithFlags")
    return (<hipError_t (*)(hipGraphExec_t*,hipGraph_t,unsigned long long) nogil> hipGraphInstantiateWithFlags_funptr)(pGraphExec,graph,flags)


cdef void* hipGraphLaunch_funptr = NULL
# @brief launches an executable graph in a stream
# @param [in] graphExec - instance of executable graph to launch.
# @param [in] stream - instance of stream in which to launch executable graph.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphLaunch(hipGraphExec_t graphExec,hipStream_t stream) nogil:
    global _lib_handle
    global hipGraphLaunch_funptr
    if hipGraphLaunch_funptr == NULL:
        with gil:
            hipGraphLaunch_funptr = loader.load_symbol(_lib_handle, "hipGraphLaunch")
    return (<hipError_t (*)(hipGraphExec_t,hipStream_t) nogil> hipGraphLaunch_funptr)(graphExec,stream)


cdef void* hipGraphUpload_funptr = NULL
# @brief uploads an executable graph in a stream
# @param [in] graphExec - instance of executable graph to launch.
# @param [in] stream - instance of stream in which to launch executable graph.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphUpload(hipGraphExec_t graphExec,hipStream_t stream) nogil:
    global _lib_handle
    global hipGraphUpload_funptr
    if hipGraphUpload_funptr == NULL:
        with gil:
            hipGraphUpload_funptr = loader.load_symbol(_lib_handle, "hipGraphUpload")
    return (<hipError_t (*)(hipGraphExec_t,hipStream_t) nogil> hipGraphUpload_funptr)(graphExec,stream)


cdef void* hipGraphExecDestroy_funptr = NULL
# @brief Destroys an executable graph
# @param [in] pGraphExec - instance of executable graph to destry.
# @returns #hipSuccess.
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) nogil:
    global _lib_handle
    global hipGraphExecDestroy_funptr
    if hipGraphExecDestroy_funptr == NULL:
        with gil:
            hipGraphExecDestroy_funptr = loader.load_symbol(_lib_handle, "hipGraphExecDestroy")
    return (<hipError_t (*)(hipGraphExec_t) nogil> hipGraphExecDestroy_funptr)(graphExec)


cdef void* hipGraphExecUpdate_funptr = NULL
# @brief Check whether an executable graph can be updated with a graph and perform the update if  *
# possible.
# @param [in] hGraphExec - instance of executable graph to update.
# @param [in] hGraph - graph that contains the updated parameters.
# @param [in] hErrorNode_out -  node which caused the permissibility check to forbid the update.
# @param [in] updateResult_out - Whether the graph update was permitted.
# @returns #hipSuccess, #hipErrorGraphExecUpdateFailure
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec,hipGraph_t hGraph,hipGraphNode_t* hErrorNode_out,hipGraphExecUpdateResult * updateResult_out) nogil:
    global _lib_handle
    global hipGraphExecUpdate_funptr
    if hipGraphExecUpdate_funptr == NULL:
        with gil:
            hipGraphExecUpdate_funptr = loader.load_symbol(_lib_handle, "hipGraphExecUpdate")
    return (<hipError_t (*)(hipGraphExec_t,hipGraph_t,hipGraphNode_t*,hipGraphExecUpdateResult *) nogil> hipGraphExecUpdate_funptr)(hGraphExec,hGraph,hErrorNode_out,updateResult_out)


cdef void* hipGraphAddKernelNode_funptr = NULL
# @brief Creates a kernel execution node and adds it to a graph.
# @param [out] pGraphNode - pointer to graph node to create.
# @param [in] graph - instance of graph to add the created node.
# @param [in] pDependencies - pointer to the dependencies on the kernel execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] pNodeParams - pointer to the parameters to the kernel execution node on the GPU.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDeviceFunction
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipKernelNodeParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphAddKernelNode_funptr
    if hipGraphAddKernelNode_funptr == NULL:
        with gil:
            hipGraphAddKernelNode_funptr = loader.load_symbol(_lib_handle, "hipGraphAddKernelNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,hipKernelNodeParams *) nogil> hipGraphAddKernelNode_funptr)(pGraphNode,graph,pDependencies,numDependencies,pNodeParams)


cdef void* hipGraphKernelNodeGetParams_funptr = NULL
# @brief Gets kernel node's parameters.
# @param [in] node - instance of the node to get parameters from.
# @param [out] pNodeParams - pointer to the parameters
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphKernelNodeGetParams_funptr
    if hipGraphKernelNodeGetParams_funptr == NULL:
        with gil:
            hipGraphKernelNodeGetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphKernelNodeGetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipKernelNodeParams *) nogil> hipGraphKernelNodeGetParams_funptr)(node,pNodeParams)


cdef void* hipGraphKernelNodeSetParams_funptr = NULL
# @brief Sets a kernel node's parameters.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - const pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphKernelNodeSetParams_funptr
    if hipGraphKernelNodeSetParams_funptr == NULL:
        with gil:
            hipGraphKernelNodeSetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphKernelNodeSetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipKernelNodeParams *) nogil> hipGraphKernelNodeSetParams_funptr)(node,pNodeParams)


cdef void* hipGraphExecKernelNodeSetParams_funptr = NULL
# @brief Sets the parameters for a kernel node in the given graphExec.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - const pointer to the kernel node parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphExecKernelNodeSetParams_funptr
    if hipGraphExecKernelNodeSetParams_funptr == NULL:
        with gil:
            hipGraphExecKernelNodeSetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphExecKernelNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipKernelNodeParams *) nogil> hipGraphExecKernelNodeSetParams_funptr)(hGraphExec,node,pNodeParams)


cdef void* hipGraphAddMemcpyNode_funptr = NULL
# @brief Creates a memcpy node and adds it to a graph.
# @param [out] pGraphNode - pointer to graph node to create.
# @param [in] graph - instance of graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] pCopyParams - const pointer to the parameters for the memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipMemcpy3DParms * pCopyParams) nogil:
    global _lib_handle
    global hipGraphAddMemcpyNode_funptr
    if hipGraphAddMemcpyNode_funptr == NULL:
        with gil:
            hipGraphAddMemcpyNode_funptr = loader.load_symbol(_lib_handle, "hipGraphAddMemcpyNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,hipMemcpy3DParms *) nogil> hipGraphAddMemcpyNode_funptr)(pGraphNode,graph,pDependencies,numDependencies,pCopyParams)


cdef void* hipGraphMemcpyNodeGetParams_funptr = NULL
# @brief Gets a memcpy node's parameters.
# @param [in] node - instance of the node to get parameters from.
# @param [out] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil:
    global _lib_handle
    global hipGraphMemcpyNodeGetParams_funptr
    if hipGraphMemcpyNodeGetParams_funptr == NULL:
        with gil:
            hipGraphMemcpyNodeGetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphMemcpyNodeGetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipMemcpy3DParms *) nogil> hipGraphMemcpyNodeGetParams_funptr)(node,pNodeParams)


cdef void* hipGraphMemcpyNodeSetParams_funptr = NULL
# @brief Sets a memcpy node's parameters.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - const pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil:
    global _lib_handle
    global hipGraphMemcpyNodeSetParams_funptr
    if hipGraphMemcpyNodeSetParams_funptr == NULL:
        with gil:
            hipGraphMemcpyNodeSetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphMemcpyNodeSetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipMemcpy3DParms *) nogil> hipGraphMemcpyNodeSetParams_funptr)(node,pNodeParams)


cdef void* hipGraphKernelNodeSetAttribute_funptr = NULL
# @brief Sets a node attribute.
# @param [in] hNode - instance of the node to set parameters to.
# @param [in] attr - the attribute node is set to.
# @param [in] value - const pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode,hipKernelNodeAttrID attr,hipKernelNodeAttrValue * value) nogil:
    global _lib_handle
    global hipGraphKernelNodeSetAttribute_funptr
    if hipGraphKernelNodeSetAttribute_funptr == NULL:
        with gil:
            hipGraphKernelNodeSetAttribute_funptr = loader.load_symbol(_lib_handle, "hipGraphKernelNodeSetAttribute")
    return (<hipError_t (*)(hipGraphNode_t,hipKernelNodeAttrID,hipKernelNodeAttrValue *) nogil> hipGraphKernelNodeSetAttribute_funptr)(hNode,attr,value)


cdef void* hipGraphKernelNodeGetAttribute_funptr = NULL
# @brief Gets a node attribute.
# @param [in] hNode - instance of the node to set parameters to.
# @param [in] attr - the attribute node is set to.
# @param [in] value - const pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode,hipKernelNodeAttrID attr,hipKernelNodeAttrValue * value) nogil:
    global _lib_handle
    global hipGraphKernelNodeGetAttribute_funptr
    if hipGraphKernelNodeGetAttribute_funptr == NULL:
        with gil:
            hipGraphKernelNodeGetAttribute_funptr = loader.load_symbol(_lib_handle, "hipGraphKernelNodeGetAttribute")
    return (<hipError_t (*)(hipGraphNode_t,hipKernelNodeAttrID,hipKernelNodeAttrValue *) nogil> hipGraphKernelNodeGetAttribute_funptr)(hNode,attr,value)


cdef void* hipGraphExecMemcpyNodeSetParams_funptr = NULL
# @brief Sets the parameters for a memcpy node in the given graphExec.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - const pointer to the kernel node parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil:
    global _lib_handle
    global hipGraphExecMemcpyNodeSetParams_funptr
    if hipGraphExecMemcpyNodeSetParams_funptr == NULL:
        with gil:
            hipGraphExecMemcpyNodeSetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphExecMemcpyNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipMemcpy3DParms *) nogil> hipGraphExecMemcpyNodeSetParams_funptr)(hGraphExec,node,pNodeParams)


cdef void* hipGraphAddMemcpyNode1D_funptr = NULL
# @brief Creates a 1D memcpy node and adds it to a graph.
# @param [out] pGraphNode - pointer to graph node to create.
# @param [in] graph - instance of graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] dst - pointer to memory address to the destination.
# @param [in] src - pointer to memory address to the source.
# @param [in] count - the size of the memory to copy.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,void * dst,const void * src,int count,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipGraphAddMemcpyNode1D_funptr
    if hipGraphAddMemcpyNode1D_funptr == NULL:
        with gil:
            hipGraphAddMemcpyNode1D_funptr = loader.load_symbol(_lib_handle, "hipGraphAddMemcpyNode1D")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,void *,const void *,int,hipMemcpyKind) nogil> hipGraphAddMemcpyNode1D_funptr)(pGraphNode,graph,pDependencies,numDependencies,dst,src,count,kind)


cdef void* hipGraphMemcpyNodeSetParams1D_funptr = NULL
# @brief Sets a memcpy node's parameters to perform a 1-dimensional copy.
# @param [in] node - instance of the node to set parameters to.
# @param [in] dst - pointer to memory address to the destination.
# @param [in] src - pointer to memory address to the source.
# @param [in] count - the size of the memory to copy.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node,void * dst,const void * src,int count,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipGraphMemcpyNodeSetParams1D_funptr
    if hipGraphMemcpyNodeSetParams1D_funptr == NULL:
        with gil:
            hipGraphMemcpyNodeSetParams1D_funptr = loader.load_symbol(_lib_handle, "hipGraphMemcpyNodeSetParams1D")
    return (<hipError_t (*)(hipGraphNode_t,void *,const void *,int,hipMemcpyKind) nogil> hipGraphMemcpyNodeSetParams1D_funptr)(node,dst,src,count,kind)


cdef void* hipGraphExecMemcpyNodeSetParams1D_funptr = NULL
# @brief Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional
# copy.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] dst - pointer to memory address to the destination.
# @param [in] src - pointer to memory address to the source.
# @param [in] count - the size of the memory to copy.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec,hipGraphNode_t node,void * dst,const void * src,int count,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipGraphExecMemcpyNodeSetParams1D_funptr
    if hipGraphExecMemcpyNodeSetParams1D_funptr == NULL:
        with gil:
            hipGraphExecMemcpyNodeSetParams1D_funptr = loader.load_symbol(_lib_handle, "hipGraphExecMemcpyNodeSetParams1D")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,int,hipMemcpyKind) nogil> hipGraphExecMemcpyNodeSetParams1D_funptr)(hGraphExec,node,dst,src,count,kind)


cdef void* hipGraphAddMemcpyNodeFromSymbol_funptr = NULL
# @brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph.
# @param [out] pGraphNode - pointer to graph node to create.
# @param [in] graph - instance of graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] dst - pointer to memory address to the destination.
# @param [in] symbol - Device symbol address.
# @param [in] count - the size of the memory to copy.
# @param [in] offset - Offset from start of symbol in bytes.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,void * dst,const void * symbol,int count,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipGraphAddMemcpyNodeFromSymbol_funptr
    if hipGraphAddMemcpyNodeFromSymbol_funptr == NULL:
        with gil:
            hipGraphAddMemcpyNodeFromSymbol_funptr = loader.load_symbol(_lib_handle, "hipGraphAddMemcpyNodeFromSymbol")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,void *,const void *,int,int,hipMemcpyKind) nogil> hipGraphAddMemcpyNodeFromSymbol_funptr)(pGraphNode,graph,pDependencies,numDependencies,dst,symbol,count,offset,kind)


cdef void* hipGraphMemcpyNodeSetParamsFromSymbol_funptr = NULL
# @brief Sets a memcpy node's parameters to copy from a symbol on the device.
# @param [in] node - instance of the node to set parameters to.
# @param [in] dst - pointer to memory address to the destination.
# @param [in] symbol - Device symbol address.
# @param [in] count - the size of the memory to copy.
# @param [in] offset - Offset from start of symbol in bytes.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node,void * dst,const void * symbol,int count,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipGraphMemcpyNodeSetParamsFromSymbol_funptr
    if hipGraphMemcpyNodeSetParamsFromSymbol_funptr == NULL:
        with gil:
            hipGraphMemcpyNodeSetParamsFromSymbol_funptr = loader.load_symbol(_lib_handle, "hipGraphMemcpyNodeSetParamsFromSymbol")
    return (<hipError_t (*)(hipGraphNode_t,void *,const void *,int,int,hipMemcpyKind) nogil> hipGraphMemcpyNodeSetParamsFromSymbol_funptr)(node,dst,symbol,count,offset,kind)


cdef void* hipGraphExecMemcpyNodeSetParamsFromSymbol_funptr = NULL
# @brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the
# device.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] dst - pointer to memory address to the destination.
# @param [in] symbol - Device symbol address.
# @param [in] count - the size of the memory to copy.
# @param [in] offset - Offset from start of symbol in bytes.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec,hipGraphNode_t node,void * dst,const void * symbol,int count,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipGraphExecMemcpyNodeSetParamsFromSymbol_funptr
    if hipGraphExecMemcpyNodeSetParamsFromSymbol_funptr == NULL:
        with gil:
            hipGraphExecMemcpyNodeSetParamsFromSymbol_funptr = loader.load_symbol(_lib_handle, "hipGraphExecMemcpyNodeSetParamsFromSymbol")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,int,int,hipMemcpyKind) nogil> hipGraphExecMemcpyNodeSetParamsFromSymbol_funptr)(hGraphExec,node,dst,symbol,count,offset,kind)


cdef void* hipGraphAddMemcpyNodeToSymbol_funptr = NULL
# @brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph.
# @param [out] pGraphNode - pointer to graph node to create.
# @param [in] graph - instance of graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] symbol - Device symbol address.
# @param [in] src - pointer to memory address of the src.
# @param [in] count - the size of the memory to copy.
# @param [in] offset - Offset from start of symbol in bytes.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,const void * symbol,const void * src,int count,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipGraphAddMemcpyNodeToSymbol_funptr
    if hipGraphAddMemcpyNodeToSymbol_funptr == NULL:
        with gil:
            hipGraphAddMemcpyNodeToSymbol_funptr = loader.load_symbol(_lib_handle, "hipGraphAddMemcpyNodeToSymbol")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,const void *,const void *,int,int,hipMemcpyKind) nogil> hipGraphAddMemcpyNodeToSymbol_funptr)(pGraphNode,graph,pDependencies,numDependencies,symbol,src,count,offset,kind)


cdef void* hipGraphMemcpyNodeSetParamsToSymbol_funptr = NULL
# @brief Sets a memcpy node's parameters to copy to a symbol on the device.
# @param [in] node - instance of the node to set parameters to.
# @param [in] symbol - Device symbol address.
# @param [in] src - pointer to memory address of the src.
# @param [in] count - the size of the memory to copy.
# @param [in] offset - Offset from start of symbol in bytes.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node,const void * symbol,const void * src,int count,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipGraphMemcpyNodeSetParamsToSymbol_funptr
    if hipGraphMemcpyNodeSetParamsToSymbol_funptr == NULL:
        with gil:
            hipGraphMemcpyNodeSetParamsToSymbol_funptr = loader.load_symbol(_lib_handle, "hipGraphMemcpyNodeSetParamsToSymbol")
    return (<hipError_t (*)(hipGraphNode_t,const void *,const void *,int,int,hipMemcpyKind) nogil> hipGraphMemcpyNodeSetParamsToSymbol_funptr)(node,symbol,src,count,offset,kind)


cdef void* hipGraphExecMemcpyNodeSetParamsToSymbol_funptr = NULL
# @brief Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the
# device.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] symbol - Device symbol address.
# @param [in] src - pointer to memory address of the src.
# @param [in] count - the size of the memory to copy.
# @param [in] offset - Offset from start of symbol in bytes.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec,hipGraphNode_t node,const void * symbol,const void * src,int count,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipGraphExecMemcpyNodeSetParamsToSymbol_funptr
    if hipGraphExecMemcpyNodeSetParamsToSymbol_funptr == NULL:
        with gil:
            hipGraphExecMemcpyNodeSetParamsToSymbol_funptr = loader.load_symbol(_lib_handle, "hipGraphExecMemcpyNodeSetParamsToSymbol")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,const void *,const void *,int,int,hipMemcpyKind) nogil> hipGraphExecMemcpyNodeSetParamsToSymbol_funptr)(hGraphExec,node,symbol,src,count,offset,kind)


cdef void* hipGraphAddMemsetNode_funptr = NULL
# @brief Creates a memset node and adds it to a graph.
# @param [out] pGraphNode - pointer to the graph node to create.
# @param [in] graph - instance of the graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] pMemsetParams - const pointer to the parameters for the memory set.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipMemsetParams * pMemsetParams) nogil:
    global _lib_handle
    global hipGraphAddMemsetNode_funptr
    if hipGraphAddMemsetNode_funptr == NULL:
        with gil:
            hipGraphAddMemsetNode_funptr = loader.load_symbol(_lib_handle, "hipGraphAddMemsetNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,hipMemsetParams *) nogil> hipGraphAddMemsetNode_funptr)(pGraphNode,graph,pDependencies,numDependencies,pMemsetParams)


cdef void* hipGraphMemsetNodeGetParams_funptr = NULL
# @brief Gets a memset node's parameters.
# @param [in] node - instane of the node to get parameters from.
# @param [out] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphMemsetNodeGetParams_funptr
    if hipGraphMemsetNodeGetParams_funptr == NULL:
        with gil:
            hipGraphMemsetNodeGetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphMemsetNodeGetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipMemsetParams *) nogil> hipGraphMemsetNodeGetParams_funptr)(node,pNodeParams)


cdef void* hipGraphMemsetNodeSetParams_funptr = NULL
# @brief Sets a memset node's parameters.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphMemsetNodeSetParams_funptr
    if hipGraphMemsetNodeSetParams_funptr == NULL:
        with gil:
            hipGraphMemsetNodeSetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphMemsetNodeSetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipMemsetParams *) nogil> hipGraphMemsetNodeSetParams_funptr)(node,pNodeParams)


cdef void* hipGraphExecMemsetNodeSetParams_funptr = NULL
# @brief Sets the parameters for a memset node in the given graphExec.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphExecMemsetNodeSetParams_funptr
    if hipGraphExecMemsetNodeSetParams_funptr == NULL:
        with gil:
            hipGraphExecMemsetNodeSetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphExecMemsetNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipMemsetParams *) nogil> hipGraphExecMemsetNodeSetParams_funptr)(hGraphExec,node,pNodeParams)


cdef void* hipGraphAddHostNode_funptr = NULL
# @brief Creates a host execution node and adds it to a graph.
# @param [out] pGraphNode - pointer to the graph node to create.
# @param [in] graph - instance of the graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] pNodeParams -pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipHostNodeParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphAddHostNode_funptr
    if hipGraphAddHostNode_funptr == NULL:
        with gil:
            hipGraphAddHostNode_funptr = loader.load_symbol(_lib_handle, "hipGraphAddHostNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,hipHostNodeParams *) nogil> hipGraphAddHostNode_funptr)(pGraphNode,graph,pDependencies,numDependencies,pNodeParams)


cdef void* hipGraphHostNodeGetParams_funptr = NULL
# @brief Returns a host node's parameters.
# @param [in] node - instane of the node to get parameters from.
# @param [out] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphHostNodeGetParams_funptr
    if hipGraphHostNodeGetParams_funptr == NULL:
        with gil:
            hipGraphHostNodeGetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphHostNodeGetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipHostNodeParams *) nogil> hipGraphHostNodeGetParams_funptr)(node,pNodeParams)


cdef void* hipGraphHostNodeSetParams_funptr = NULL
# @brief Sets a host node's parameters.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphHostNodeSetParams_funptr
    if hipGraphHostNodeSetParams_funptr == NULL:
        with gil:
            hipGraphHostNodeSetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphHostNodeSetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipHostNodeParams *) nogil> hipGraphHostNodeSetParams_funptr)(node,pNodeParams)


cdef void* hipGraphExecHostNodeSetParams_funptr = NULL
# @brief Sets the parameters for a host node in the given graphExec.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil:
    global _lib_handle
    global hipGraphExecHostNodeSetParams_funptr
    if hipGraphExecHostNodeSetParams_funptr == NULL:
        with gil:
            hipGraphExecHostNodeSetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphExecHostNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipHostNodeParams *) nogil> hipGraphExecHostNodeSetParams_funptr)(hGraphExec,node,pNodeParams)


cdef void* hipGraphAddChildGraphNode_funptr = NULL
# @brief Creates a child graph node and adds it to a graph.
# @param [out] pGraphNode - pointer to the graph node to create.
# @param [in] graph - instance of the graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] childGraph - the graph to clone into this node
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipGraph_t childGraph) nogil:
    global _lib_handle
    global hipGraphAddChildGraphNode_funptr
    if hipGraphAddChildGraphNode_funptr == NULL:
        with gil:
            hipGraphAddChildGraphNode_funptr = loader.load_symbol(_lib_handle, "hipGraphAddChildGraphNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,hipGraph_t) nogil> hipGraphAddChildGraphNode_funptr)(pGraphNode,graph,pDependencies,numDependencies,childGraph)


cdef void* hipGraphChildGraphNodeGetGraph_funptr = NULL
# @brief Gets a handle to the embedded graph of a child graph node.
# @param [in] node - instane of the node to get child graph.
# @param [out] pGraph - pointer to get the graph.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node,hipGraph_t* pGraph) nogil:
    global _lib_handle
    global hipGraphChildGraphNodeGetGraph_funptr
    if hipGraphChildGraphNodeGetGraph_funptr == NULL:
        with gil:
            hipGraphChildGraphNodeGetGraph_funptr = loader.load_symbol(_lib_handle, "hipGraphChildGraphNodeGetGraph")
    return (<hipError_t (*)(hipGraphNode_t,hipGraph_t*) nogil> hipGraphChildGraphNodeGetGraph_funptr)(node,pGraph)


cdef void* hipGraphExecChildGraphNodeSetParams_funptr = NULL
# @brief Updates node parameters in the child graph node in the given graphExec.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - node from the graph which was used to instantiate graphExec.
# @param [in] childGraph - child graph with updated parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipGraph_t childGraph) nogil:
    global _lib_handle
    global hipGraphExecChildGraphNodeSetParams_funptr
    if hipGraphExecChildGraphNodeSetParams_funptr == NULL:
        with gil:
            hipGraphExecChildGraphNodeSetParams_funptr = loader.load_symbol(_lib_handle, "hipGraphExecChildGraphNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipGraph_t) nogil> hipGraphExecChildGraphNodeSetParams_funptr)(hGraphExec,node,childGraph)


cdef void* hipGraphAddEmptyNode_funptr = NULL
# @brief Creates an empty node and adds it to a graph.
# @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
# @param [in] graph - instane of the graph the node is add to.
# @param [in] pDependencies - const pointer to the node dependenties.
# @param [in] numDependencies - the number of dependencies.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies) nogil:
    global _lib_handle
    global hipGraphAddEmptyNode_funptr
    if hipGraphAddEmptyNode_funptr == NULL:
        with gil:
            hipGraphAddEmptyNode_funptr = loader.load_symbol(_lib_handle, "hipGraphAddEmptyNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int) nogil> hipGraphAddEmptyNode_funptr)(pGraphNode,graph,pDependencies,numDependencies)


cdef void* hipGraphAddEventRecordNode_funptr = NULL
# @brief Creates an event record node and adds it to a graph.
# @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
# @param [in] graph - instane of the graph the node to be added.
# @param [in] pDependencies - const pointer to the node dependenties.
# @param [in] numDependencies - the number of dependencies.
# @param [in] event - Event for the node.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipEvent_t event) nogil:
    global _lib_handle
    global hipGraphAddEventRecordNode_funptr
    if hipGraphAddEventRecordNode_funptr == NULL:
        with gil:
            hipGraphAddEventRecordNode_funptr = loader.load_symbol(_lib_handle, "hipGraphAddEventRecordNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,hipEvent_t) nogil> hipGraphAddEventRecordNode_funptr)(pGraphNode,graph,pDependencies,numDependencies,event)


cdef void* hipGraphEventRecordNodeGetEvent_funptr = NULL
# @brief Returns the event associated with an event record node.
# @param [in] node -  instane of the node to get event from.
# @param [out] event_out - Pointer to return the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node,hipEvent_t* event_out) nogil:
    global _lib_handle
    global hipGraphEventRecordNodeGetEvent_funptr
    if hipGraphEventRecordNodeGetEvent_funptr == NULL:
        with gil:
            hipGraphEventRecordNodeGetEvent_funptr = loader.load_symbol(_lib_handle, "hipGraphEventRecordNodeGetEvent")
    return (<hipError_t (*)(hipGraphNode_t,hipEvent_t*) nogil> hipGraphEventRecordNodeGetEvent_funptr)(node,event_out)


cdef void* hipGraphEventRecordNodeSetEvent_funptr = NULL
# @brief Sets an event record node's event.
# @param [in] node - instane of the node to set event to.
# @param [in] event - pointer to the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node,hipEvent_t event) nogil:
    global _lib_handle
    global hipGraphEventRecordNodeSetEvent_funptr
    if hipGraphEventRecordNodeSetEvent_funptr == NULL:
        with gil:
            hipGraphEventRecordNodeSetEvent_funptr = loader.load_symbol(_lib_handle, "hipGraphEventRecordNodeSetEvent")
    return (<hipError_t (*)(hipGraphNode_t,hipEvent_t) nogil> hipGraphEventRecordNodeSetEvent_funptr)(node,event)


cdef void* hipGraphExecEventRecordNodeSetEvent_funptr = NULL
# @brief Sets the event for an event record node in the given graphExec.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] hNode - node from the graph which was used to instantiate graphExec.
# @param [in] event - pointer to the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec,hipGraphNode_t hNode,hipEvent_t event) nogil:
    global _lib_handle
    global hipGraphExecEventRecordNodeSetEvent_funptr
    if hipGraphExecEventRecordNodeSetEvent_funptr == NULL:
        with gil:
            hipGraphExecEventRecordNodeSetEvent_funptr = loader.load_symbol(_lib_handle, "hipGraphExecEventRecordNodeSetEvent")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t) nogil> hipGraphExecEventRecordNodeSetEvent_funptr)(hGraphExec,hNode,event)


cdef void* hipGraphAddEventWaitNode_funptr = NULL
# @brief Creates an event wait node and adds it to a graph.
# @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
# @param [in] graph - instane of the graph the node to be added.
# @param [in] pDependencies - const pointer to the node dependenties.
# @param [in] numDependencies - the number of dependencies.
# @param [in] event - Event for the node.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipEvent_t event) nogil:
    global _lib_handle
    global hipGraphAddEventWaitNode_funptr
    if hipGraphAddEventWaitNode_funptr == NULL:
        with gil:
            hipGraphAddEventWaitNode_funptr = loader.load_symbol(_lib_handle, "hipGraphAddEventWaitNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,int,hipEvent_t) nogil> hipGraphAddEventWaitNode_funptr)(pGraphNode,graph,pDependencies,numDependencies,event)


cdef void* hipGraphEventWaitNodeGetEvent_funptr = NULL
# @brief Returns the event associated with an event wait node.
# @param [in] node -  instane of the node to get event from.
# @param [out] event_out - Pointer to return the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node,hipEvent_t* event_out) nogil:
    global _lib_handle
    global hipGraphEventWaitNodeGetEvent_funptr
    if hipGraphEventWaitNodeGetEvent_funptr == NULL:
        with gil:
            hipGraphEventWaitNodeGetEvent_funptr = loader.load_symbol(_lib_handle, "hipGraphEventWaitNodeGetEvent")
    return (<hipError_t (*)(hipGraphNode_t,hipEvent_t*) nogil> hipGraphEventWaitNodeGetEvent_funptr)(node,event_out)


cdef void* hipGraphEventWaitNodeSetEvent_funptr = NULL
# @brief Sets an event wait node's event.
# @param [in] node - instane of the node to set event to.
# @param [in] event - pointer to the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node,hipEvent_t event) nogil:
    global _lib_handle
    global hipGraphEventWaitNodeSetEvent_funptr
    if hipGraphEventWaitNodeSetEvent_funptr == NULL:
        with gil:
            hipGraphEventWaitNodeSetEvent_funptr = loader.load_symbol(_lib_handle, "hipGraphEventWaitNodeSetEvent")
    return (<hipError_t (*)(hipGraphNode_t,hipEvent_t) nogil> hipGraphEventWaitNodeSetEvent_funptr)(node,event)


cdef void* hipGraphExecEventWaitNodeSetEvent_funptr = NULL
# @brief Sets the event for an event record node in the given graphExec.
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] hNode - node from the graph which was used to instantiate graphExec.
# @param [in] event - pointer to the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec,hipGraphNode_t hNode,hipEvent_t event) nogil:
    global _lib_handle
    global hipGraphExecEventWaitNodeSetEvent_funptr
    if hipGraphExecEventWaitNodeSetEvent_funptr == NULL:
        with gil:
            hipGraphExecEventWaitNodeSetEvent_funptr = loader.load_symbol(_lib_handle, "hipGraphExecEventWaitNodeSetEvent")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t) nogil> hipGraphExecEventWaitNodeSetEvent_funptr)(hGraphExec,hNode,event)


cdef void* hipDeviceGetGraphMemAttribute_funptr = NULL
# @brief Get the mem attribute for graphs.
# @param [in] device - device the attr is get for.
# @param [in] attr - attr to get.
# @param [out] value - value for specific attr.
# @returns #hipSuccess, #hipErrorInvalidDevice
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceGetGraphMemAttribute(int device,hipGraphMemAttributeType attr,void * value) nogil:
    global _lib_handle
    global hipDeviceGetGraphMemAttribute_funptr
    if hipDeviceGetGraphMemAttribute_funptr == NULL:
        with gil:
            hipDeviceGetGraphMemAttribute_funptr = loader.load_symbol(_lib_handle, "hipDeviceGetGraphMemAttribute")
    return (<hipError_t (*)(int,hipGraphMemAttributeType,void *) nogil> hipDeviceGetGraphMemAttribute_funptr)(device,attr,value)


cdef void* hipDeviceSetGraphMemAttribute_funptr = NULL
# @brief Set the mem attribute for graphs.
# @param [in] device - device the attr is set for.
# @param [in] attr - attr to set.
# @param [in] value - value for specific attr.
# @returns #hipSuccess, #hipErrorInvalidDevice
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceSetGraphMemAttribute(int device,hipGraphMemAttributeType attr,void * value) nogil:
    global _lib_handle
    global hipDeviceSetGraphMemAttribute_funptr
    if hipDeviceSetGraphMemAttribute_funptr == NULL:
        with gil:
            hipDeviceSetGraphMemAttribute_funptr = loader.load_symbol(_lib_handle, "hipDeviceSetGraphMemAttribute")
    return (<hipError_t (*)(int,hipGraphMemAttributeType,void *) nogil> hipDeviceSetGraphMemAttribute_funptr)(device,attr,value)


cdef void* hipDeviceGraphMemTrim_funptr = NULL
# @brief Free unused memory on specific device used for graph back to OS.
# @param [in] device - device the memory is used for graphs
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceGraphMemTrim(int device) nogil:
    global _lib_handle
    global hipDeviceGraphMemTrim_funptr
    if hipDeviceGraphMemTrim_funptr == NULL:
        with gil:
            hipDeviceGraphMemTrim_funptr = loader.load_symbol(_lib_handle, "hipDeviceGraphMemTrim")
    return (<hipError_t (*)(int) nogil> hipDeviceGraphMemTrim_funptr)(device)


cdef void* hipUserObjectCreate_funptr = NULL
# @brief Create an instance of userObject to manage lifetime of a resource.
# @param [out] object_out - pointer to instace of userobj.
# @param [in] ptr - pointer to pass to destroy function.
# @param [in] destroy - destroy callback to remove resource.
# @param [in] initialRefcount - reference to resource.
# @param [in] flags - flags passed to API.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipUserObjectCreate(hipUserObject_t* object_out,void * ptr,hipHostFn_t destroy,unsigned int initialRefcount,unsigned int flags) nogil:
    global _lib_handle
    global hipUserObjectCreate_funptr
    if hipUserObjectCreate_funptr == NULL:
        with gil:
            hipUserObjectCreate_funptr = loader.load_symbol(_lib_handle, "hipUserObjectCreate")
    return (<hipError_t (*)(hipUserObject_t*,void *,hipHostFn_t,unsigned int,unsigned int) nogil> hipUserObjectCreate_funptr)(object_out,ptr,destroy,initialRefcount,flags)


cdef void* hipUserObjectRelease_funptr = NULL
# @brief Release number of references to resource.
# @param [in] object - pointer to instace of userobj.
# @param [in] count - reference to resource to be retained.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipUserObjectRelease(hipUserObject_t object,unsigned int count) nogil:
    global _lib_handle
    global hipUserObjectRelease_funptr
    if hipUserObjectRelease_funptr == NULL:
        with gil:
            hipUserObjectRelease_funptr = loader.load_symbol(_lib_handle, "hipUserObjectRelease")
    return (<hipError_t (*)(hipUserObject_t,unsigned int) nogil> hipUserObjectRelease_funptr)(object,count)


cdef void* hipUserObjectRetain_funptr = NULL
# @brief Retain number of references to resource.
# @param [in] object - pointer to instace of userobj.
# @param [in] count - reference to resource to be retained.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipUserObjectRetain(hipUserObject_t object,unsigned int count) nogil:
    global _lib_handle
    global hipUserObjectRetain_funptr
    if hipUserObjectRetain_funptr == NULL:
        with gil:
            hipUserObjectRetain_funptr = loader.load_symbol(_lib_handle, "hipUserObjectRetain")
    return (<hipError_t (*)(hipUserObject_t,unsigned int) nogil> hipUserObjectRetain_funptr)(object,count)


cdef void* hipGraphRetainUserObject_funptr = NULL
# @brief Retain user object for graphs.
# @param [in] graph - pointer to graph to retain the user object for.
# @param [in] object - pointer to instace of userobj.
# @param [in] count - reference to resource to be retained.
# @param [in] flags - flags passed to API.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphRetainUserObject(hipGraph_t graph,hipUserObject_t object,unsigned int count,unsigned int flags) nogil:
    global _lib_handle
    global hipGraphRetainUserObject_funptr
    if hipGraphRetainUserObject_funptr == NULL:
        with gil:
            hipGraphRetainUserObject_funptr = loader.load_symbol(_lib_handle, "hipGraphRetainUserObject")
    return (<hipError_t (*)(hipGraph_t,hipUserObject_t,unsigned int,unsigned int) nogil> hipGraphRetainUserObject_funptr)(graph,object,count,flags)


cdef void* hipGraphReleaseUserObject_funptr = NULL
# @brief Release user object from graphs.
# @param [in] graph - pointer to graph to retain the user object for.
# @param [in] object - pointer to instace of userobj.
# @param [in] count - reference to resource to be retained.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphReleaseUserObject(hipGraph_t graph,hipUserObject_t object,unsigned int count) nogil:
    global _lib_handle
    global hipGraphReleaseUserObject_funptr
    if hipGraphReleaseUserObject_funptr == NULL:
        with gil:
            hipGraphReleaseUserObject_funptr = loader.load_symbol(_lib_handle, "hipGraphReleaseUserObject")
    return (<hipError_t (*)(hipGraph_t,hipUserObject_t,unsigned int) nogil> hipGraphReleaseUserObject_funptr)(graph,object,count)


cdef void* hipMemAddressFree_funptr = NULL
# @brief Frees an address range reservation made via hipMemAddressReserve
# @param [in] devPtr - starting address of the range.
# @param [in] size - size of the range.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemAddressFree(void * devPtr,int size) nogil:
    global _lib_handle
    global hipMemAddressFree_funptr
    if hipMemAddressFree_funptr == NULL:
        with gil:
            hipMemAddressFree_funptr = loader.load_symbol(_lib_handle, "hipMemAddressFree")
    return (<hipError_t (*)(void *,int) nogil> hipMemAddressFree_funptr)(devPtr,size)


cdef void* hipMemAddressReserve_funptr = NULL
# @brief Reserves an address range
# @param [out] ptr - starting address of the reserved range.
# @param [in] size - size of the reservation.
# @param [in] alignment - alignment of the address.
# @param [in] addr - requested starting address of the range.
# @param [in] flags - currently unused, must be zero.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemAddressReserve(void ** ptr,int size,int alignment,void * addr,unsigned long long flags) nogil:
    global _lib_handle
    global hipMemAddressReserve_funptr
    if hipMemAddressReserve_funptr == NULL:
        with gil:
            hipMemAddressReserve_funptr = loader.load_symbol(_lib_handle, "hipMemAddressReserve")
    return (<hipError_t (*)(void **,int,int,void *,unsigned long long) nogil> hipMemAddressReserve_funptr)(ptr,size,alignment,addr,flags)


cdef void* hipMemCreate_funptr = NULL
# @brief Creates a memory allocation described by the properties and size
# @param [out] handle - value of the returned handle.
# @param [in] size - size of the allocation.
# @param [in] prop - properties of the allocation.
# @param [in] flags - currently unused, must be zero.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle,int size,hipMemAllocationProp * prop,unsigned long long flags) nogil:
    global _lib_handle
    global hipMemCreate_funptr
    if hipMemCreate_funptr == NULL:
        with gil:
            hipMemCreate_funptr = loader.load_symbol(_lib_handle, "hipMemCreate")
    return (<hipError_t (*)(hipMemGenericAllocationHandle_t*,int,hipMemAllocationProp *,unsigned long long) nogil> hipMemCreate_funptr)(handle,size,prop,flags)


cdef void* hipMemExportToShareableHandle_funptr = NULL
# @brief Exports an allocation to a requested shareable handle type.
# @param [out] shareableHandle - value of the returned handle.
# @param [in] handle - handle to share.
# @param [in] handleType - type of the shareable handle.
# @param [in] flags - currently unused, must be zero.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemExportToShareableHandle(void * shareableHandle,hipMemGenericAllocationHandle_t handle,hipMemAllocationHandleType handleType,unsigned long long flags) nogil:
    global _lib_handle
    global hipMemExportToShareableHandle_funptr
    if hipMemExportToShareableHandle_funptr == NULL:
        with gil:
            hipMemExportToShareableHandle_funptr = loader.load_symbol(_lib_handle, "hipMemExportToShareableHandle")
    return (<hipError_t (*)(void *,hipMemGenericAllocationHandle_t,hipMemAllocationHandleType,unsigned long long) nogil> hipMemExportToShareableHandle_funptr)(shareableHandle,handle,handleType,flags)


cdef void* hipMemGetAccess_funptr = NULL
# @brief Get the access flags set for the given location and ptr.
# @param [out] flags - flags for this location.
# @param [in] location - target location.
# @param [in] ptr - address to check the access flags.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemGetAccess(unsigned long long * flags,hipMemLocation * location,void * ptr) nogil:
    global _lib_handle
    global hipMemGetAccess_funptr
    if hipMemGetAccess_funptr == NULL:
        with gil:
            hipMemGetAccess_funptr = loader.load_symbol(_lib_handle, "hipMemGetAccess")
    return (<hipError_t (*)(unsigned long long *,hipMemLocation *,void *) nogil> hipMemGetAccess_funptr)(flags,location,ptr)


cdef void* hipMemGetAllocationGranularity_funptr = NULL
# @brief Calculates either the minimal or recommended granularity.
# @param [out] granularity - returned granularity.
# @param [in] prop - location properties.
# @param [in] option - determines which granularity to return.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemGetAllocationGranularity(int * granularity,hipMemAllocationProp * prop,hipMemAllocationGranularity_flags option) nogil:
    global _lib_handle
    global hipMemGetAllocationGranularity_funptr
    if hipMemGetAllocationGranularity_funptr == NULL:
        with gil:
            hipMemGetAllocationGranularity_funptr = loader.load_symbol(_lib_handle, "hipMemGetAllocationGranularity")
    return (<hipError_t (*)(int *,hipMemAllocationProp *,hipMemAllocationGranularity_flags) nogil> hipMemGetAllocationGranularity_funptr)(granularity,prop,option)


cdef void* hipMemGetAllocationPropertiesFromHandle_funptr = NULL
# @brief Retrieve the property structure of the given handle.
# @param [out] prop - properties of the given handle.
# @param [in] handle - handle to perform the query on.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp * prop,hipMemGenericAllocationHandle_t handle) nogil:
    global _lib_handle
    global hipMemGetAllocationPropertiesFromHandle_funptr
    if hipMemGetAllocationPropertiesFromHandle_funptr == NULL:
        with gil:
            hipMemGetAllocationPropertiesFromHandle_funptr = loader.load_symbol(_lib_handle, "hipMemGetAllocationPropertiesFromHandle")
    return (<hipError_t (*)(hipMemAllocationProp *,hipMemGenericAllocationHandle_t) nogil> hipMemGetAllocationPropertiesFromHandle_funptr)(prop,handle)


cdef void* hipMemImportFromShareableHandle_funptr = NULL
# @brief Imports an allocation from a requested shareable handle type.
# @param [out] handle - returned value.
# @param [in] osHandle - shareable handle representing the memory allocation.
# @param [in] shHandleType - handle type.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle,void * osHandle,hipMemAllocationHandleType shHandleType) nogil:
    global _lib_handle
    global hipMemImportFromShareableHandle_funptr
    if hipMemImportFromShareableHandle_funptr == NULL:
        with gil:
            hipMemImportFromShareableHandle_funptr = loader.load_symbol(_lib_handle, "hipMemImportFromShareableHandle")
    return (<hipError_t (*)(hipMemGenericAllocationHandle_t*,void *,hipMemAllocationHandleType) nogil> hipMemImportFromShareableHandle_funptr)(handle,osHandle,shHandleType)


cdef void* hipMemMap_funptr = NULL
# @brief Maps an allocation handle to a reserved virtual address range.
# @param [in] ptr - address where the memory will be mapped.
# @param [in] size - size of the mapping.
# @param [in] offset - offset into the memory, currently must be zero.
# @param [in] handle - memory allocation to be mapped.
# @param [in] flags - currently unused, must be zero.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemMap(void * ptr,int size,int offset,hipMemGenericAllocationHandle_t handle,unsigned long long flags) nogil:
    global _lib_handle
    global hipMemMap_funptr
    if hipMemMap_funptr == NULL:
        with gil:
            hipMemMap_funptr = loader.load_symbol(_lib_handle, "hipMemMap")
    return (<hipError_t (*)(void *,int,int,hipMemGenericAllocationHandle_t,unsigned long long) nogil> hipMemMap_funptr)(ptr,size,offset,handle,flags)


cdef void* hipMemMapArrayAsync_funptr = NULL
# @brief Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.
# @param [in] mapInfoList - list of hipArrayMapInfo.
# @param [in] count - number of hipArrayMapInfo in mapInfoList.
# @param [in] stream - stream identifier for the stream to use for map or unmap operations.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemMapArrayAsync(hipArrayMapInfo * mapInfoList,unsigned int count,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemMapArrayAsync_funptr
    if hipMemMapArrayAsync_funptr == NULL:
        with gil:
            hipMemMapArrayAsync_funptr = loader.load_symbol(_lib_handle, "hipMemMapArrayAsync")
    return (<hipError_t (*)(hipArrayMapInfo *,unsigned int,hipStream_t) nogil> hipMemMapArrayAsync_funptr)(mapInfoList,count,stream)


cdef void* hipMemRelease_funptr = NULL
# @brief Release a memory handle representing a memory allocation which was previously allocated through hipMemCreate.
# @param [in] handle - handle of the memory allocation.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) nogil:
    global _lib_handle
    global hipMemRelease_funptr
    if hipMemRelease_funptr == NULL:
        with gil:
            hipMemRelease_funptr = loader.load_symbol(_lib_handle, "hipMemRelease")
    return (<hipError_t (*)(hipMemGenericAllocationHandle_t) nogil> hipMemRelease_funptr)(handle)


cdef void* hipMemRetainAllocationHandle_funptr = NULL
# @brief Returns the allocation handle of the backing memory allocation given the address.
# @param [out] handle - handle representing addr.
# @param [in] addr - address to look up.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle,void * addr) nogil:
    global _lib_handle
    global hipMemRetainAllocationHandle_funptr
    if hipMemRetainAllocationHandle_funptr == NULL:
        with gil:
            hipMemRetainAllocationHandle_funptr = loader.load_symbol(_lib_handle, "hipMemRetainAllocationHandle")
    return (<hipError_t (*)(hipMemGenericAllocationHandle_t*,void *) nogil> hipMemRetainAllocationHandle_funptr)(handle,addr)


cdef void* hipMemSetAccess_funptr = NULL
# @brief Set the access flags for each location specified in desc for the given virtual address range.
# @param [in] ptr - starting address of the virtual address range.
# @param [in] size - size of the range.
# @param [in] desc - array of hipMemAccessDesc.
# @param [in] count - number of hipMemAccessDesc in desc.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemSetAccess(void * ptr,int size,hipMemAccessDesc * desc,int count) nogil:
    global _lib_handle
    global hipMemSetAccess_funptr
    if hipMemSetAccess_funptr == NULL:
        with gil:
            hipMemSetAccess_funptr = loader.load_symbol(_lib_handle, "hipMemSetAccess")
    return (<hipError_t (*)(void *,int,hipMemAccessDesc *,int) nogil> hipMemSetAccess_funptr)(ptr,size,desc,count)


cdef void* hipMemUnmap_funptr = NULL
# @brief Unmap memory allocation of a given address range.
# @param [in] ptr - starting address of the range to unmap.
# @param [in] size - size of the virtual address range.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemUnmap(void * ptr,int size) nogil:
    global _lib_handle
    global hipMemUnmap_funptr
    if hipMemUnmap_funptr == NULL:
        with gil:
            hipMemUnmap_funptr = loader.load_symbol(_lib_handle, "hipMemUnmap")
    return (<hipError_t (*)(void *,int) nogil> hipMemUnmap_funptr)(ptr,size)


cdef void* hipGLGetDevices_funptr = NULL
cdef hipError_t hipGLGetDevices(unsigned int * pHipDeviceCount,int * pHipDevices,unsigned int hipDeviceCount,hipGLDeviceList deviceList) nogil:
    global _lib_handle
    global hipGLGetDevices_funptr
    if hipGLGetDevices_funptr == NULL:
        with gil:
            hipGLGetDevices_funptr = loader.load_symbol(_lib_handle, "hipGLGetDevices")
    return (<hipError_t (*)(unsigned int *,int *,unsigned int,hipGLDeviceList) nogil> hipGLGetDevices_funptr)(pHipDeviceCount,pHipDevices,hipDeviceCount,deviceList)


cdef void* hipGraphicsGLRegisterBuffer_funptr = NULL
cdef hipError_t hipGraphicsGLRegisterBuffer(_hipGraphicsResource ** resource,GLuint buffer,unsigned int flags) nogil:
    global _lib_handle
    global hipGraphicsGLRegisterBuffer_funptr
    if hipGraphicsGLRegisterBuffer_funptr == NULL:
        with gil:
            hipGraphicsGLRegisterBuffer_funptr = loader.load_symbol(_lib_handle, "hipGraphicsGLRegisterBuffer")
    return (<hipError_t (*)(_hipGraphicsResource **,GLuint,unsigned int) nogil> hipGraphicsGLRegisterBuffer_funptr)(resource,buffer,flags)


cdef void* hipGraphicsGLRegisterImage_funptr = NULL
cdef hipError_t hipGraphicsGLRegisterImage(_hipGraphicsResource ** resource,GLuint image,GLenum target,unsigned int flags) nogil:
    global _lib_handle
    global hipGraphicsGLRegisterImage_funptr
    if hipGraphicsGLRegisterImage_funptr == NULL:
        with gil:
            hipGraphicsGLRegisterImage_funptr = loader.load_symbol(_lib_handle, "hipGraphicsGLRegisterImage")
    return (<hipError_t (*)(_hipGraphicsResource **,GLuint,GLenum,unsigned int) nogil> hipGraphicsGLRegisterImage_funptr)(resource,image,target,flags)


cdef void* hipGraphicsMapResources_funptr = NULL
cdef hipError_t hipGraphicsMapResources(int count,hipGraphicsResource_t* resources,hipStream_t stream) nogil:
    global _lib_handle
    global hipGraphicsMapResources_funptr
    if hipGraphicsMapResources_funptr == NULL:
        with gil:
            hipGraphicsMapResources_funptr = loader.load_symbol(_lib_handle, "hipGraphicsMapResources")
    return (<hipError_t (*)(int,hipGraphicsResource_t*,hipStream_t) nogil> hipGraphicsMapResources_funptr)(count,resources,stream)


cdef void* hipGraphicsSubResourceGetMappedArray_funptr = NULL
cdef hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array,hipGraphicsResource_t resource,unsigned int arrayIndex,unsigned int mipLevel) nogil:
    global _lib_handle
    global hipGraphicsSubResourceGetMappedArray_funptr
    if hipGraphicsSubResourceGetMappedArray_funptr == NULL:
        with gil:
            hipGraphicsSubResourceGetMappedArray_funptr = loader.load_symbol(_lib_handle, "hipGraphicsSubResourceGetMappedArray")
    return (<hipError_t (*)(hipArray_t*,hipGraphicsResource_t,unsigned int,unsigned int) nogil> hipGraphicsSubResourceGetMappedArray_funptr)(array,resource,arrayIndex,mipLevel)


cdef void* hipGraphicsResourceGetMappedPointer_funptr = NULL
cdef hipError_t hipGraphicsResourceGetMappedPointer(void ** devPtr,int * size,hipGraphicsResource_t resource) nogil:
    global _lib_handle
    global hipGraphicsResourceGetMappedPointer_funptr
    if hipGraphicsResourceGetMappedPointer_funptr == NULL:
        with gil:
            hipGraphicsResourceGetMappedPointer_funptr = loader.load_symbol(_lib_handle, "hipGraphicsResourceGetMappedPointer")
    return (<hipError_t (*)(void **,int *,hipGraphicsResource_t) nogil> hipGraphicsResourceGetMappedPointer_funptr)(devPtr,size,resource)


cdef void* hipGraphicsUnmapResources_funptr = NULL
cdef hipError_t hipGraphicsUnmapResources(int count,hipGraphicsResource_t* resources,hipStream_t stream) nogil:
    global _lib_handle
    global hipGraphicsUnmapResources_funptr
    if hipGraphicsUnmapResources_funptr == NULL:
        with gil:
            hipGraphicsUnmapResources_funptr = loader.load_symbol(_lib_handle, "hipGraphicsUnmapResources")
    return (<hipError_t (*)(int,hipGraphicsResource_t*,hipStream_t) nogil> hipGraphicsUnmapResources_funptr)(count,resources,stream)


cdef void* hipGraphicsUnregisterResource_funptr = NULL
cdef hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) nogil:
    global _lib_handle
    global hipGraphicsUnregisterResource_funptr
    if hipGraphicsUnregisterResource_funptr == NULL:
        with gil:
            hipGraphicsUnregisterResource_funptr = loader.load_symbol(_lib_handle, "hipGraphicsUnregisterResource")
    return (<hipError_t (*)(hipGraphicsResource_t) nogil> hipGraphicsUnregisterResource_funptr)(resource)


cdef void* hipMemcpy_spt_funptr = NULL
cdef hipError_t hipMemcpy_spt(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpy_spt_funptr
    if hipMemcpy_spt_funptr == NULL:
        with gil:
            hipMemcpy_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpy_spt")
    return (<hipError_t (*)(void *,const void *,int,hipMemcpyKind) nogil> hipMemcpy_spt_funptr)(dst,src,sizeBytes,kind)


cdef void* hipMemcpyToSymbol_spt_funptr = NULL
cdef hipError_t hipMemcpyToSymbol_spt(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpyToSymbol_spt_funptr
    if hipMemcpyToSymbol_spt_funptr == NULL:
        with gil:
            hipMemcpyToSymbol_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpyToSymbol_spt")
    return (<hipError_t (*)(const void *,const void *,int,int,hipMemcpyKind) nogil> hipMemcpyToSymbol_spt_funptr)(symbol,src,sizeBytes,offset,kind)


cdef void* hipMemcpyFromSymbol_spt_funptr = NULL
cdef hipError_t hipMemcpyFromSymbol_spt(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpyFromSymbol_spt_funptr
    if hipMemcpyFromSymbol_spt_funptr == NULL:
        with gil:
            hipMemcpyFromSymbol_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpyFromSymbol_spt")
    return (<hipError_t (*)(void *,const void *,int,int,hipMemcpyKind) nogil> hipMemcpyFromSymbol_spt_funptr)(dst,symbol,sizeBytes,offset,kind)


cdef void* hipMemcpy2D_spt_funptr = NULL
cdef hipError_t hipMemcpy2D_spt(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpy2D_spt_funptr
    if hipMemcpy2D_spt_funptr == NULL:
        with gil:
            hipMemcpy2D_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2D_spt")
    return (<hipError_t (*)(void *,int,const void *,int,int,int,hipMemcpyKind) nogil> hipMemcpy2D_spt_funptr)(dst,dpitch,src,spitch,width,height,kind)


cdef void* hipMemcpy2DFromArray_spt_funptr = NULL
cdef hipError_t hipMemcpy2DFromArray_spt(void * dst,int dpitch,hipArray_const_t src,int wOffset,int hOffset,int width,int height,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpy2DFromArray_spt_funptr
    if hipMemcpy2DFromArray_spt_funptr == NULL:
        with gil:
            hipMemcpy2DFromArray_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DFromArray_spt")
    return (<hipError_t (*)(void *,int,hipArray_const_t,int,int,int,int,hipMemcpyKind) nogil> hipMemcpy2DFromArray_spt_funptr)(dst,dpitch,src,wOffset,hOffset,width,height,kind)


cdef void* hipMemcpy3D_spt_funptr = NULL
cdef hipError_t hipMemcpy3D_spt(hipMemcpy3DParms * p) nogil:
    global _lib_handle
    global hipMemcpy3D_spt_funptr
    if hipMemcpy3D_spt_funptr == NULL:
        with gil:
            hipMemcpy3D_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpy3D_spt")
    return (<hipError_t (*)(hipMemcpy3DParms *) nogil> hipMemcpy3D_spt_funptr)(p)


cdef void* hipMemset_spt_funptr = NULL
cdef hipError_t hipMemset_spt(void * dst,int value,int sizeBytes) nogil:
    global _lib_handle
    global hipMemset_spt_funptr
    if hipMemset_spt_funptr == NULL:
        with gil:
            hipMemset_spt_funptr = loader.load_symbol(_lib_handle, "hipMemset_spt")
    return (<hipError_t (*)(void *,int,int) nogil> hipMemset_spt_funptr)(dst,value,sizeBytes)


cdef void* hipMemsetAsync_spt_funptr = NULL
cdef hipError_t hipMemsetAsync_spt(void * dst,int value,int sizeBytes,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemsetAsync_spt_funptr
    if hipMemsetAsync_spt_funptr == NULL:
        with gil:
            hipMemsetAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemsetAsync_spt")
    return (<hipError_t (*)(void *,int,int,hipStream_t) nogil> hipMemsetAsync_spt_funptr)(dst,value,sizeBytes,stream)


cdef void* hipMemset2D_spt_funptr = NULL
cdef hipError_t hipMemset2D_spt(void * dst,int pitch,int value,int width,int height) nogil:
    global _lib_handle
    global hipMemset2D_spt_funptr
    if hipMemset2D_spt_funptr == NULL:
        with gil:
            hipMemset2D_spt_funptr = loader.load_symbol(_lib_handle, "hipMemset2D_spt")
    return (<hipError_t (*)(void *,int,int,int,int) nogil> hipMemset2D_spt_funptr)(dst,pitch,value,width,height)


cdef void* hipMemset2DAsync_spt_funptr = NULL
cdef hipError_t hipMemset2DAsync_spt(void * dst,int pitch,int value,int width,int height,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemset2DAsync_spt_funptr
    if hipMemset2DAsync_spt_funptr == NULL:
        with gil:
            hipMemset2DAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemset2DAsync_spt")
    return (<hipError_t (*)(void *,int,int,int,int,hipStream_t) nogil> hipMemset2DAsync_spt_funptr)(dst,pitch,value,width,height,stream)


cdef void* hipMemset3DAsync_spt_funptr = NULL
cdef hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemset3DAsync_spt_funptr
    if hipMemset3DAsync_spt_funptr == NULL:
        with gil:
            hipMemset3DAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemset3DAsync_spt")
    return (<hipError_t (*)(hipPitchedPtr,int,hipExtent,hipStream_t) nogil> hipMemset3DAsync_spt_funptr)(pitchedDevPtr,value,extent,stream)


cdef void* hipMemset3D_spt_funptr = NULL
cdef hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent) nogil:
    global _lib_handle
    global hipMemset3D_spt_funptr
    if hipMemset3D_spt_funptr == NULL:
        with gil:
            hipMemset3D_spt_funptr = loader.load_symbol(_lib_handle, "hipMemset3D_spt")
    return (<hipError_t (*)(hipPitchedPtr,int,hipExtent) nogil> hipMemset3D_spt_funptr)(pitchedDevPtr,value,extent)


cdef void* hipMemcpyAsync_spt_funptr = NULL
cdef hipError_t hipMemcpyAsync_spt(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyAsync_spt_funptr
    if hipMemcpyAsync_spt_funptr == NULL:
        with gil:
            hipMemcpyAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpyAsync_spt")
    return (<hipError_t (*)(void *,const void *,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpyAsync_spt_funptr)(dst,src,sizeBytes,kind,stream)


cdef void* hipMemcpy3DAsync_spt_funptr = NULL
cdef hipError_t hipMemcpy3DAsync_spt(hipMemcpy3DParms * p,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpy3DAsync_spt_funptr
    if hipMemcpy3DAsync_spt_funptr == NULL:
        with gil:
            hipMemcpy3DAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpy3DAsync_spt")
    return (<hipError_t (*)(hipMemcpy3DParms *,hipStream_t) nogil> hipMemcpy3DAsync_spt_funptr)(p,stream)


cdef void* hipMemcpy2DAsync_spt_funptr = NULL
cdef hipError_t hipMemcpy2DAsync_spt(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpy2DAsync_spt_funptr
    if hipMemcpy2DAsync_spt_funptr == NULL:
        with gil:
            hipMemcpy2DAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DAsync_spt")
    return (<hipError_t (*)(void *,int,const void *,int,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpy2DAsync_spt_funptr)(dst,dpitch,src,spitch,width,height,kind,stream)


cdef void* hipMemcpyFromSymbolAsync_spt_funptr = NULL
cdef hipError_t hipMemcpyFromSymbolAsync_spt(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyFromSymbolAsync_spt_funptr
    if hipMemcpyFromSymbolAsync_spt_funptr == NULL:
        with gil:
            hipMemcpyFromSymbolAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpyFromSymbolAsync_spt")
    return (<hipError_t (*)(void *,const void *,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpyFromSymbolAsync_spt_funptr)(dst,symbol,sizeBytes,offset,kind,stream)


cdef void* hipMemcpyToSymbolAsync_spt_funptr = NULL
cdef hipError_t hipMemcpyToSymbolAsync_spt(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpyToSymbolAsync_spt_funptr
    if hipMemcpyToSymbolAsync_spt_funptr == NULL:
        with gil:
            hipMemcpyToSymbolAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpyToSymbolAsync_spt")
    return (<hipError_t (*)(const void *,const void *,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpyToSymbolAsync_spt_funptr)(symbol,src,sizeBytes,offset,kind,stream)


cdef void* hipMemcpyFromArray_spt_funptr = NULL
cdef hipError_t hipMemcpyFromArray_spt(void * dst,hipArray_const_t src,int wOffsetSrc,int hOffset,int count,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpyFromArray_spt_funptr
    if hipMemcpyFromArray_spt_funptr == NULL:
        with gil:
            hipMemcpyFromArray_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpyFromArray_spt")
    return (<hipError_t (*)(void *,hipArray_const_t,int,int,int,hipMemcpyKind) nogil> hipMemcpyFromArray_spt_funptr)(dst,src,wOffsetSrc,hOffset,count,kind)


cdef void* hipMemcpy2DToArray_spt_funptr = NULL
cdef hipError_t hipMemcpy2DToArray_spt(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil:
    global _lib_handle
    global hipMemcpy2DToArray_spt_funptr
    if hipMemcpy2DToArray_spt_funptr == NULL:
        with gil:
            hipMemcpy2DToArray_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DToArray_spt")
    return (<hipError_t (*)(hipArray *,int,int,const void *,int,int,int,hipMemcpyKind) nogil> hipMemcpy2DToArray_spt_funptr)(dst,wOffset,hOffset,src,spitch,width,height,kind)


cdef void* hipMemcpy2DFromArrayAsync_spt_funptr = NULL
cdef hipError_t hipMemcpy2DFromArrayAsync_spt(void * dst,int dpitch,hipArray_const_t src,int wOffsetSrc,int hOffsetSrc,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpy2DFromArrayAsync_spt_funptr
    if hipMemcpy2DFromArrayAsync_spt_funptr == NULL:
        with gil:
            hipMemcpy2DFromArrayAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DFromArrayAsync_spt")
    return (<hipError_t (*)(void *,int,hipArray_const_t,int,int,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpy2DFromArrayAsync_spt_funptr)(dst,dpitch,src,wOffsetSrc,hOffsetSrc,width,height,kind,stream)


cdef void* hipMemcpy2DToArrayAsync_spt_funptr = NULL
cdef hipError_t hipMemcpy2DToArrayAsync_spt(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _lib_handle
    global hipMemcpy2DToArrayAsync_spt_funptr
    if hipMemcpy2DToArrayAsync_spt_funptr == NULL:
        with gil:
            hipMemcpy2DToArrayAsync_spt_funptr = loader.load_symbol(_lib_handle, "hipMemcpy2DToArrayAsync_spt")
    return (<hipError_t (*)(hipArray *,int,int,const void *,int,int,int,hipMemcpyKind,hipStream_t) nogil> hipMemcpy2DToArrayAsync_spt_funptr)(dst,wOffset,hOffset,src,spitch,width,height,kind,stream)


cdef void* hipStreamQuery_spt_funptr = NULL
cdef hipError_t hipStreamQuery_spt(hipStream_t stream) nogil:
    global _lib_handle
    global hipStreamQuery_spt_funptr
    if hipStreamQuery_spt_funptr == NULL:
        with gil:
            hipStreamQuery_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamQuery_spt")
    return (<hipError_t (*)(hipStream_t) nogil> hipStreamQuery_spt_funptr)(stream)


cdef void* hipStreamSynchronize_spt_funptr = NULL
cdef hipError_t hipStreamSynchronize_spt(hipStream_t stream) nogil:
    global _lib_handle
    global hipStreamSynchronize_spt_funptr
    if hipStreamSynchronize_spt_funptr == NULL:
        with gil:
            hipStreamSynchronize_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamSynchronize_spt")
    return (<hipError_t (*)(hipStream_t) nogil> hipStreamSynchronize_spt_funptr)(stream)


cdef void* hipStreamGetPriority_spt_funptr = NULL
cdef hipError_t hipStreamGetPriority_spt(hipStream_t stream,int * priority) nogil:
    global _lib_handle
    global hipStreamGetPriority_spt_funptr
    if hipStreamGetPriority_spt_funptr == NULL:
        with gil:
            hipStreamGetPriority_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamGetPriority_spt")
    return (<hipError_t (*)(hipStream_t,int *) nogil> hipStreamGetPriority_spt_funptr)(stream,priority)


cdef void* hipStreamWaitEvent_spt_funptr = NULL
cdef hipError_t hipStreamWaitEvent_spt(hipStream_t stream,hipEvent_t event,unsigned int flags) nogil:
    global _lib_handle
    global hipStreamWaitEvent_spt_funptr
    if hipStreamWaitEvent_spt_funptr == NULL:
        with gil:
            hipStreamWaitEvent_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamWaitEvent_spt")
    return (<hipError_t (*)(hipStream_t,hipEvent_t,unsigned int) nogil> hipStreamWaitEvent_spt_funptr)(stream,event,flags)


cdef void* hipStreamGetFlags_spt_funptr = NULL
cdef hipError_t hipStreamGetFlags_spt(hipStream_t stream,unsigned int * flags) nogil:
    global _lib_handle
    global hipStreamGetFlags_spt_funptr
    if hipStreamGetFlags_spt_funptr == NULL:
        with gil:
            hipStreamGetFlags_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamGetFlags_spt")
    return (<hipError_t (*)(hipStream_t,unsigned int *) nogil> hipStreamGetFlags_spt_funptr)(stream,flags)


cdef void* hipStreamAddCallback_spt_funptr = NULL
cdef hipError_t hipStreamAddCallback_spt(hipStream_t stream,hipStreamCallback_t callback,void * userData,unsigned int flags) nogil:
    global _lib_handle
    global hipStreamAddCallback_spt_funptr
    if hipStreamAddCallback_spt_funptr == NULL:
        with gil:
            hipStreamAddCallback_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamAddCallback_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCallback_t,void *,unsigned int) nogil> hipStreamAddCallback_spt_funptr)(stream,callback,userData,flags)


cdef void* hipEventRecord_spt_funptr = NULL
cdef hipError_t hipEventRecord_spt(hipEvent_t event,hipStream_t stream) nogil:
    global _lib_handle
    global hipEventRecord_spt_funptr
    if hipEventRecord_spt_funptr == NULL:
        with gil:
            hipEventRecord_spt_funptr = loader.load_symbol(_lib_handle, "hipEventRecord_spt")
    return (<hipError_t (*)(hipEvent_t,hipStream_t) nogil> hipEventRecord_spt_funptr)(event,stream)


cdef void* hipLaunchCooperativeKernel_spt_funptr = NULL
cdef hipError_t hipLaunchCooperativeKernel_spt(const void * f,dim3 gridDim,dim3 blockDim,void ** kernelParams,uint32_t sharedMemBytes,hipStream_t hStream) nogil:
    global _lib_handle
    global hipLaunchCooperativeKernel_spt_funptr
    if hipLaunchCooperativeKernel_spt_funptr == NULL:
        with gil:
            hipLaunchCooperativeKernel_spt_funptr = loader.load_symbol(_lib_handle, "hipLaunchCooperativeKernel_spt")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,uint32_t,hipStream_t) nogil> hipLaunchCooperativeKernel_spt_funptr)(f,gridDim,blockDim,kernelParams,sharedMemBytes,hStream)


cdef void* hipLaunchKernel_spt_funptr = NULL
cdef hipError_t hipLaunchKernel_spt(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,int sharedMemBytes,hipStream_t stream) nogil:
    global _lib_handle
    global hipLaunchKernel_spt_funptr
    if hipLaunchKernel_spt_funptr == NULL:
        with gil:
            hipLaunchKernel_spt_funptr = loader.load_symbol(_lib_handle, "hipLaunchKernel_spt")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,int,hipStream_t) nogil> hipLaunchKernel_spt_funptr)(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream)


cdef void* hipGraphLaunch_spt_funptr = NULL
cdef hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec,hipStream_t stream) nogil:
    global _lib_handle
    global hipGraphLaunch_spt_funptr
    if hipGraphLaunch_spt_funptr == NULL:
        with gil:
            hipGraphLaunch_spt_funptr = loader.load_symbol(_lib_handle, "hipGraphLaunch_spt")
    return (<hipError_t (*)(hipGraphExec_t,hipStream_t) nogil> hipGraphLaunch_spt_funptr)(graphExec,stream)


cdef void* hipStreamBeginCapture_spt_funptr = NULL
cdef hipError_t hipStreamBeginCapture_spt(hipStream_t stream,hipStreamCaptureMode mode) nogil:
    global _lib_handle
    global hipStreamBeginCapture_spt_funptr
    if hipStreamBeginCapture_spt_funptr == NULL:
        with gil:
            hipStreamBeginCapture_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamBeginCapture_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureMode) nogil> hipStreamBeginCapture_spt_funptr)(stream,mode)


cdef void* hipStreamEndCapture_spt_funptr = NULL
cdef hipError_t hipStreamEndCapture_spt(hipStream_t stream,hipGraph_t* pGraph) nogil:
    global _lib_handle
    global hipStreamEndCapture_spt_funptr
    if hipStreamEndCapture_spt_funptr == NULL:
        with gil:
            hipStreamEndCapture_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamEndCapture_spt")
    return (<hipError_t (*)(hipStream_t,hipGraph_t*) nogil> hipStreamEndCapture_spt_funptr)(stream,pGraph)


cdef void* hipStreamIsCapturing_spt_funptr = NULL
cdef hipError_t hipStreamIsCapturing_spt(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus) nogil:
    global _lib_handle
    global hipStreamIsCapturing_spt_funptr
    if hipStreamIsCapturing_spt_funptr == NULL:
        with gil:
            hipStreamIsCapturing_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamIsCapturing_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *) nogil> hipStreamIsCapturing_spt_funptr)(stream,pCaptureStatus)


cdef void* hipStreamGetCaptureInfo_spt_funptr = NULL
cdef hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus,unsigned long long * pId) nogil:
    global _lib_handle
    global hipStreamGetCaptureInfo_spt_funptr
    if hipStreamGetCaptureInfo_spt_funptr == NULL:
        with gil:
            hipStreamGetCaptureInfo_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamGetCaptureInfo_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *) nogil> hipStreamGetCaptureInfo_spt_funptr)(stream,pCaptureStatus,pId)


cdef void* hipStreamGetCaptureInfo_v2_spt_funptr = NULL
cdef hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,hipStreamCaptureStatus * captureStatus_out,unsigned long long * id_out,hipGraph_t* graph_out,hipGraphNode_t ** dependencies_out,int * numDependencies_out) nogil:
    global _lib_handle
    global hipStreamGetCaptureInfo_v2_spt_funptr
    if hipStreamGetCaptureInfo_v2_spt_funptr == NULL:
        with gil:
            hipStreamGetCaptureInfo_v2_spt_funptr = loader.load_symbol(_lib_handle, "hipStreamGetCaptureInfo_v2_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *,hipGraph_t*,hipGraphNode_t **,int *) nogil> hipStreamGetCaptureInfo_v2_spt_funptr)(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out)


cdef void* hipLaunchHostFunc_spt_funptr = NULL
cdef hipError_t hipLaunchHostFunc_spt(hipStream_t stream,hipHostFn_t fn,void * userData) nogil:
    global _lib_handle
    global hipLaunchHostFunc_spt_funptr
    if hipLaunchHostFunc_spt_funptr == NULL:
        with gil:
            hipLaunchHostFunc_spt_funptr = loader.load_symbol(_lib_handle, "hipLaunchHostFunc_spt")
    return (<hipError_t (*)(hipStream_t,hipHostFn_t,void *) nogil> hipLaunchHostFunc_spt_funptr)(stream,fn,userData)
