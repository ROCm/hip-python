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
cimport hip._util.posixloader as loader
cdef void* _lib_handle = NULL

cdef void __init() nogil:
    global _lib_handle
    if _lib_handle == NULL:
        with gil:
            _lib_handle = loader.open_library("libamdhip64.so")

cdef void __init_symbol(void** result, const char* name) nogil:
    global _lib_handle
    if _lib_handle == NULL:
        __init()
    if result[0] == NULL:
        with gil:
            result[0] = loader.load_symbol(_lib_handle, name) 


cdef void* _hipCreateChannelDesc__funptr = NULL
cdef hipChannelFormatDesc hipCreateChannelDesc(int x,int y,int z,int w,hipChannelFormatKind f) nogil:
    global _hipCreateChannelDesc__funptr
    __init_symbol(&_hipCreateChannelDesc__funptr,"hipCreateChannelDesc")
    return (<hipChannelFormatDesc (*)(int,int,int,int,hipChannelFormatKind) nogil> _hipCreateChannelDesc__funptr)(x,y,z,w,f)


cdef void* _hipInit__funptr = NULL
# 
#  @defgroup API HIP API
#  @{
# 
#  Defines the HIP API.  See the individual sections for more information.
# 
# 
#  @defgroup Driver Initialization and Version
#  @{
#  This section describes the initializtion and version functions of HIP runtime API.
# 
# 
# 
# @brief Explicitly initializes the HIP runtime.
# 
# Most HIP APIs implicitly initialize the HIP runtime.
# This API provides control over the timing of the initialization.
cdef hipError_t hipInit(unsigned int flags) nogil:
    global _hipInit__funptr
    __init_symbol(&_hipInit__funptr,"hipInit")
    return (<hipError_t (*)(unsigned int) nogil> _hipInit__funptr)(flags)


cdef void* _hipDriverGetVersion__funptr = NULL
# 
# @brief Returns the approximate HIP driver version.
# 
# @param [out] driverVersion
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning The HIP feature set does not correspond to an exact CUDA SDK driver revision.
# This function always set *driverVersion to 4 as an approximation though HIP supports
# some features which were introduced in later CUDA SDK revisions.
# HIP apps code should not rely on the driver revision number here and should
# use arch feature flags to test device capabilities or conditional compilation.
# 
# @see hipRuntimeGetVersion
cdef hipError_t hipDriverGetVersion(int * driverVersion) nogil:
    global _hipDriverGetVersion__funptr
    __init_symbol(&_hipDriverGetVersion__funptr,"hipDriverGetVersion")
    return (<hipError_t (*)(int *) nogil> _hipDriverGetVersion__funptr)(driverVersion)


cdef void* _hipRuntimeGetVersion__funptr = NULL
# 
# @brief Returns the approximate HIP Runtime version.
# 
# @param [out] runtimeVersion
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning The version definition of HIP runtime is different from CUDA.
# On AMD platform, the function returns HIP runtime version,
# while on NVIDIA platform, it returns CUDA runtime version.
# And there is no mapping/correlation between HIP version and CUDA version.
# 
# @see hipDriverGetVersion
cdef hipError_t hipRuntimeGetVersion(int * runtimeVersion) nogil:
    global _hipRuntimeGetVersion__funptr
    __init_symbol(&_hipRuntimeGetVersion__funptr,"hipRuntimeGetVersion")
    return (<hipError_t (*)(int *) nogil> _hipRuntimeGetVersion__funptr)(runtimeVersion)


cdef void* _hipDeviceGet__funptr = NULL
# 
# @brief Returns a handle to a compute device
# @param [out] device
# @param [in] ordinal
# 
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceGet(int * device,int ordinal) nogil:
    global _hipDeviceGet__funptr
    __init_symbol(&_hipDeviceGet__funptr,"hipDeviceGet")
    return (<hipError_t (*)(int *,int) nogil> _hipDeviceGet__funptr)(device,ordinal)


cdef void* _hipDeviceComputeCapability__funptr = NULL
# 
# @brief Returns the compute capability of the device
# @param [out] major
# @param [out] minor
# @param [in] device
# 
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceComputeCapability(int * major,int * minor,int device) nogil:
    global _hipDeviceComputeCapability__funptr
    __init_symbol(&_hipDeviceComputeCapability__funptr,"hipDeviceComputeCapability")
    return (<hipError_t (*)(int *,int *,int) nogil> _hipDeviceComputeCapability__funptr)(major,minor,device)


cdef void* _hipDeviceGetName__funptr = NULL
# 
# @brief Returns an identifer string for the device.
# @param [out] name
# @param [in] len
# @param [in] device
# 
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceGetName(char * name,int len,int device) nogil:
    global _hipDeviceGetName__funptr
    __init_symbol(&_hipDeviceGetName__funptr,"hipDeviceGetName")
    return (<hipError_t (*)(char *,int,int) nogil> _hipDeviceGetName__funptr)(name,len,device)


cdef void* _hipDeviceGetUuid__funptr = NULL
# 
# @brief Returns an UUID for the device.[BETA]
# @param [out] uuid
# @param [in] device
# 
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# 
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotInitialized,
# #hipErrorDeinitialized
cdef hipError_t hipDeviceGetUuid(hipUUID_t * uuid,int device) nogil:
    global _hipDeviceGetUuid__funptr
    __init_symbol(&_hipDeviceGetUuid__funptr,"hipDeviceGetUuid")
    return (<hipError_t (*)(hipUUID_t *,int) nogil> _hipDeviceGetUuid__funptr)(uuid,device)


cdef void* _hipDeviceGetP2PAttribute__funptr = NULL
# 
# @brief Returns a value for attr of link between two devices
# @param [out] value
# @param [in] attr
# @param [in] srcDevice
# @param [in] dstDevice
# 
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceGetP2PAttribute(int * value,hipDeviceP2PAttr attr,int srcDevice,int dstDevice) nogil:
    global _hipDeviceGetP2PAttribute__funptr
    __init_symbol(&_hipDeviceGetP2PAttribute__funptr,"hipDeviceGetP2PAttribute")
    return (<hipError_t (*)(int *,hipDeviceP2PAttr,int,int) nogil> _hipDeviceGetP2PAttribute__funptr)(value,attr,srcDevice,dstDevice)


cdef void* _hipDeviceGetPCIBusId__funptr = NULL
# 
# @brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
# @param [out] pciBusId
# @param [in] len
# @param [in] device
# 
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceGetPCIBusId(char * pciBusId,int len,int device) nogil:
    global _hipDeviceGetPCIBusId__funptr
    __init_symbol(&_hipDeviceGetPCIBusId__funptr,"hipDeviceGetPCIBusId")
    return (<hipError_t (*)(char *,int,int) nogil> _hipDeviceGetPCIBusId__funptr)(pciBusId,len,device)


cdef void* _hipDeviceGetByPCIBusId__funptr = NULL
# 
# @brief Returns a handle to a compute device.
# @param [out] device handle
# @param [in] PCI Bus ID
# 
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
cdef hipError_t hipDeviceGetByPCIBusId(int * device,const char * pciBusId) nogil:
    global _hipDeviceGetByPCIBusId__funptr
    __init_symbol(&_hipDeviceGetByPCIBusId__funptr,"hipDeviceGetByPCIBusId")
    return (<hipError_t (*)(int *,const char *) nogil> _hipDeviceGetByPCIBusId__funptr)(device,pciBusId)


cdef void* _hipDeviceTotalMem__funptr = NULL
# 
# @brief Returns the total amount of memory on the device.
# @param [out] bytes
# @param [in] device
# 
# @returns #hipSuccess, #hipErrorInvalidDevice
cdef hipError_t hipDeviceTotalMem(unsigned long * bytes,int device) nogil:
    global _hipDeviceTotalMem__funptr
    __init_symbol(&_hipDeviceTotalMem__funptr,"hipDeviceTotalMem")
    return (<hipError_t (*)(unsigned long *,int) nogil> _hipDeviceTotalMem__funptr)(bytes,device)


cdef void* _hipDeviceSynchronize__funptr = NULL
# 
# @}
# 
# 
#  @defgroup Device Device Management
#  @{
#  This section describes the device management functions of HIP runtime API.
# 
# 
# @brief Waits on all active streams on current device
# 
# When this command is invoked, the host thread gets blocked until all the commands associated
# with streams associated with the device. HIP does not support multiple blocking modes (yet!).
# 
# @returns #hipSuccess
# 
# @see hipSetDevice, hipDeviceReset
cdef hipError_t hipDeviceSynchronize() nogil:
    global _hipDeviceSynchronize__funptr
    __init_symbol(&_hipDeviceSynchronize__funptr,"hipDeviceSynchronize")
    return (<hipError_t (*)() nogil> _hipDeviceSynchronize__funptr)()


cdef void* _hipDeviceReset__funptr = NULL
# 
# @brief The state of current device is discarded and updated to a fresh state.
# 
# Calling this function deletes all streams created, memory allocated, kernels running, events
# created. Make sure that no other thread is using the device or streams, memory, kernels, events
# associated with the current device.
# 
# @returns #hipSuccess
# 
# @see hipDeviceSynchronize
cdef hipError_t hipDeviceReset() nogil:
    global _hipDeviceReset__funptr
    __init_symbol(&_hipDeviceReset__funptr,"hipDeviceReset")
    return (<hipError_t (*)() nogil> _hipDeviceReset__funptr)()


cdef void* _hipSetDevice__funptr = NULL
# 
# @brief Set default device to be used for subsequent hip API calls from this thread.
# 
# @param[in] deviceId Valid device in range 0...hipGetDeviceCount().
# 
# Sets @p device as the default device for the calling host thread.  Valid device id's are 0...
# (hipGetDeviceCount()-1).
# 
# Many HIP APIs implicitly use the "default device" :
# 
# - Any device memory subsequently allocated from this host thread (using hipMalloc) will be
# allocated on device.
# - Any streams or events created from this host thread will be associated with device.
# - Any kernels launched from this host thread (using hipLaunchKernel) will be executed on device
# (unless a specific stream is specified, in which case the device associated with that stream will
# be used).
# 
# This function may be called from any host thread.  Multiple host threads may use the same device.
# This function does no synchronization with the previous or new device, and has very little
# runtime overhead. Applications can use hipSetDevice to quickly switch the default device before
# making a HIP runtime call which uses the default device.
# 
# The default device is stored in thread-local-storage for each thread.
# Thread-pool implementations may inherit the default device of the previous thread.  A good
# practice is to always call hipSetDevice at the start of HIP coding sequency to establish a known
# standard device.
# 
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorDeviceAlreadyInUse
# 
# @see hipGetDevice, hipGetDeviceCount
cdef hipError_t hipSetDevice(int deviceId) nogil:
    global _hipSetDevice__funptr
    __init_symbol(&_hipSetDevice__funptr,"hipSetDevice")
    return (<hipError_t (*)(int) nogil> _hipSetDevice__funptr)(deviceId)


cdef void* _hipGetDevice__funptr = NULL
# 
# @brief Return the default device id for the calling host thread.
# 
# @param [out] device *device is written with the default device
# 
# HIP maintains an default device for each thread using thread-local-storage.
# This device is used implicitly for HIP runtime APIs called by this thread.
# hipGetDevice returns in * @p device the default device for the calling host thread.
# 
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @see hipSetDevice, hipGetDevicesizeBytes
cdef hipError_t hipGetDevice(int * deviceId) nogil:
    global _hipGetDevice__funptr
    __init_symbol(&_hipGetDevice__funptr,"hipGetDevice")
    return (<hipError_t (*)(int *) nogil> _hipGetDevice__funptr)(deviceId)


cdef void* _hipGetDeviceCount__funptr = NULL
# 
# @brief Return number of compute-capable devices.
# 
# @param [output] count Returns number of compute-capable devices.
# 
# @returns #hipSuccess, #hipErrorNoDevice
# 
# 
# Returns in @p *count the number of devices that have ability to run compute commands.  If there
# are no such devices, then @ref hipGetDeviceCount will return #hipErrorNoDevice. If 1 or more
# devices can be found, then hipGetDeviceCount returns #hipSuccess.
cdef hipError_t hipGetDeviceCount(int * count) nogil:
    global _hipGetDeviceCount__funptr
    __init_symbol(&_hipGetDeviceCount__funptr,"hipGetDeviceCount")
    return (<hipError_t (*)(int *) nogil> _hipGetDeviceCount__funptr)(count)


cdef void* _hipDeviceGetAttribute__funptr = NULL
# 
# @brief Query for a specific device attribute.
# 
# @param [out] pi pointer to value to return
# @param [in] attr attribute to query
# @param [in] deviceId which device to query for information
# 
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
cdef hipError_t hipDeviceGetAttribute(int * pi,hipDeviceAttribute_t attr,int deviceId) nogil:
    global _hipDeviceGetAttribute__funptr
    __init_symbol(&_hipDeviceGetAttribute__funptr,"hipDeviceGetAttribute")
    return (<hipError_t (*)(int *,hipDeviceAttribute_t,int) nogil> _hipDeviceGetAttribute__funptr)(pi,attr,deviceId)


cdef void* _hipDeviceGetDefaultMemPool__funptr = NULL
# 
# @brief Returns the default memory pool of the specified device
# 
# @param [out] mem_pool Default memory pool to return
# @param [in] device    Device index for query the default memory pool
# 
# @returns #chipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotSupported
# 
# @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool,int device) nogil:
    global _hipDeviceGetDefaultMemPool__funptr
    __init_symbol(&_hipDeviceGetDefaultMemPool__funptr,"hipDeviceGetDefaultMemPool")
    return (<hipError_t (*)(hipMemPool_t*,int) nogil> _hipDeviceGetDefaultMemPool__funptr)(mem_pool,device)


cdef void* _hipDeviceSetMemPool__funptr = NULL
# 
# @brief Sets the current memory pool of a device
# 
# The memory pool must be local to the specified device.
# @p hipMallocAsync allocates from the current mempool of the provided stream's device.
# By default, a device's current memory pool is its default memory pool.
# 
# @note Use @p hipMallocFromPoolAsync for asynchronous memory allocations from a device
# different than the one the stream runs on.
# 
# @param [in] device   Device index for the update
# @param [in] mem_pool Memory pool for update as the current on the specified device
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice, #hipErrorNotSupported
# 
# @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceSetMemPool(int device,hipMemPool_t mem_pool) nogil:
    global _hipDeviceSetMemPool__funptr
    __init_symbol(&_hipDeviceSetMemPool__funptr,"hipDeviceSetMemPool")
    return (<hipError_t (*)(int,hipMemPool_t) nogil> _hipDeviceSetMemPool__funptr)(device,mem_pool)


cdef void* _hipDeviceGetMemPool__funptr = NULL
# 
# @brief Gets the current memory pool for the specified device
# 
# Returns the last pool provided to @p hipDeviceSetMemPool for this device
# or the device's default memory pool if @p hipDeviceSetMemPool has never been called.
# By default the current mempool is the default mempool for a device,
# otherwise the returned pool must have been set with @p hipDeviceSetMemPool.
# 
# @param [out] mem_pool Current memory pool on the specified device
# @param [in] device    Device index to query the current memory pool
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# 
# @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool,int device) nogil:
    global _hipDeviceGetMemPool__funptr
    __init_symbol(&_hipDeviceGetMemPool__funptr,"hipDeviceGetMemPool")
    return (<hipError_t (*)(hipMemPool_t*,int) nogil> _hipDeviceGetMemPool__funptr)(mem_pool,device)


cdef void* _hipGetDeviceProperties__funptr = NULL
# 
# @brief Returns device properties.
# 
# @param [out] prop written with device properties
# @param [in]  deviceId which device to query for information
# 
# @return #hipSuccess, #hipErrorInvalidDevice
# @bug HCC always returns 0 for maxThreadsPerMultiProcessor
# @bug HCC always returns 0 for regsPerBlock
# @bug HCC always returns 0 for l2CacheSize
# 
# Populates hipGetDeviceProperties with information for the specified device.
cdef hipError_t hipGetDeviceProperties(hipDeviceProp_t * prop,int deviceId) nogil:
    global _hipGetDeviceProperties__funptr
    __init_symbol(&_hipGetDeviceProperties__funptr,"hipGetDeviceProperties")
    return (<hipError_t (*)(hipDeviceProp_t *,int) nogil> _hipGetDeviceProperties__funptr)(prop,deviceId)


cdef void* _hipDeviceSetCacheConfig__funptr = NULL
# 
# @brief Set L1/Shared cache partition.
# 
# @param [in] cacheConfig
# 
# @returns #hipSuccess, #hipErrorNotInitialized
# Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
# on those architectures.
#
cdef hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) nogil:
    global _hipDeviceSetCacheConfig__funptr
    __init_symbol(&_hipDeviceSetCacheConfig__funptr,"hipDeviceSetCacheConfig")
    return (<hipError_t (*)(hipFuncCache_t) nogil> _hipDeviceSetCacheConfig__funptr)(cacheConfig)


cdef void* _hipDeviceGetCacheConfig__funptr = NULL
# 
# @brief Get Cache configuration for a specific Device
# 
# @param [out] cacheConfig
# 
# @returns #hipSuccess, #hipErrorNotInitialized
# Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
# on those architectures.
#
cdef hipError_t hipDeviceGetCacheConfig(hipFuncCache_t * cacheConfig) nogil:
    global _hipDeviceGetCacheConfig__funptr
    __init_symbol(&_hipDeviceGetCacheConfig__funptr,"hipDeviceGetCacheConfig")
    return (<hipError_t (*)(hipFuncCache_t *) nogil> _hipDeviceGetCacheConfig__funptr)(cacheConfig)


cdef void* _hipDeviceGetLimit__funptr = NULL
# 
# @brief Get Resource limits of current device
# 
# @param [out] pValue
# @param [in]  limit
# 
# @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
# Note: Currently, only hipLimitMallocHeapSize is available
#
cdef hipError_t hipDeviceGetLimit(unsigned long * pValue,hipLimit_t limit) nogil:
    global _hipDeviceGetLimit__funptr
    __init_symbol(&_hipDeviceGetLimit__funptr,"hipDeviceGetLimit")
    return (<hipError_t (*)(unsigned long *,hipLimit_t) nogil> _hipDeviceGetLimit__funptr)(pValue,limit)


cdef void* _hipDeviceSetLimit__funptr = NULL
# 
# @brief Set Resource limits of current device
# 
# @param [in] limit
# @param [in] value
# 
# @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
#
cdef hipError_t hipDeviceSetLimit(hipLimit_t limit,unsigned long value) nogil:
    global _hipDeviceSetLimit__funptr
    __init_symbol(&_hipDeviceSetLimit__funptr,"hipDeviceSetLimit")
    return (<hipError_t (*)(hipLimit_t,unsigned long) nogil> _hipDeviceSetLimit__funptr)(limit,value)


cdef void* _hipDeviceGetSharedMemConfig__funptr = NULL
# 
# @brief Returns bank width of shared memory for current device
# 
# @param [out] pConfig
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
# 
# Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
#
cdef hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig * pConfig) nogil:
    global _hipDeviceGetSharedMemConfig__funptr
    __init_symbol(&_hipDeviceGetSharedMemConfig__funptr,"hipDeviceGetSharedMemConfig")
    return (<hipError_t (*)(hipSharedMemConfig *) nogil> _hipDeviceGetSharedMemConfig__funptr)(pConfig)


cdef void* _hipGetDeviceFlags__funptr = NULL
# 
# @brief Gets the flags set for current device
# 
# @param [out] flags
# 
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
cdef hipError_t hipGetDeviceFlags(unsigned int * flags) nogil:
    global _hipGetDeviceFlags__funptr
    __init_symbol(&_hipGetDeviceFlags__funptr,"hipGetDeviceFlags")
    return (<hipError_t (*)(unsigned int *) nogil> _hipGetDeviceFlags__funptr)(flags)


cdef void* _hipDeviceSetSharedMemConfig__funptr = NULL
# 
# @brief The bank width of shared memory on current device is set
# 
# @param [in] config
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
# 
# Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
#
cdef hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) nogil:
    global _hipDeviceSetSharedMemConfig__funptr
    __init_symbol(&_hipDeviceSetSharedMemConfig__funptr,"hipDeviceSetSharedMemConfig")
    return (<hipError_t (*)(hipSharedMemConfig) nogil> _hipDeviceSetSharedMemConfig__funptr)(config)


cdef void* _hipSetDeviceFlags__funptr = NULL
# 
# @brief The current device behavior is changed according the flags passed.
# 
# @param [in] flags
# 
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
# 
# 
# hipDeviceMapHost              : Allow mapping host memory.  On ROCM, this is always allowed and
# the flag is ignored. hipDeviceLmemResizeToMax      : @warning ROCm silently ignores this flag.
# 
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorSetOnActiveProcess
# 
#
cdef hipError_t hipSetDeviceFlags(unsigned int flags) nogil:
    global _hipSetDeviceFlags__funptr
    __init_symbol(&_hipSetDeviceFlags__funptr,"hipSetDeviceFlags")
    return (<hipError_t (*)(unsigned int) nogil> _hipSetDeviceFlags__funptr)(flags)


cdef void* _hipChooseDevice__funptr = NULL
# 
# @brief Device which matches hipDeviceProp_t is returned
# 
# @param [out] device ID
# @param [in]  device properties pointer
# 
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipChooseDevice(int * device,hipDeviceProp_t * prop) nogil:
    global _hipChooseDevice__funptr
    __init_symbol(&_hipChooseDevice__funptr,"hipChooseDevice")
    return (<hipError_t (*)(int *,hipDeviceProp_t *) nogil> _hipChooseDevice__funptr)(device,prop)


cdef void* _hipExtGetLinkTypeAndHopCount__funptr = NULL
# 
# @brief Returns the link type and hop count between two devices
# 
# @param [in] device1 Ordinal for device1
# @param [in] device2 Ordinal for device2
# @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
# @param [out] hopcount Returns the hop count between the two devices
# 
# Queries and returns the HSA link type and the hop count between the two specified devices.
# 
# @returns #hipSuccess, #hipInvalidDevice, #hipErrorRuntimeOther
cdef hipError_t hipExtGetLinkTypeAndHopCount(int device1,int device2,unsigned int * linktype,unsigned int * hopcount) nogil:
    global _hipExtGetLinkTypeAndHopCount__funptr
    __init_symbol(&_hipExtGetLinkTypeAndHopCount__funptr,"hipExtGetLinkTypeAndHopCount")
    return (<hipError_t (*)(int,int,unsigned int *,unsigned int *) nogil> _hipExtGetLinkTypeAndHopCount__funptr)(device1,device2,linktype,hopcount)


cdef void* _hipIpcGetMemHandle__funptr = NULL
# 
# @brief Gets an interprocess memory handle for an existing device memory
#          allocation
# 
# Takes a pointer to the base of an existing device memory allocation created
# with hipMalloc and exports it for use in another process. This is a
# lightweight operation and may be called multiple times on an allocation
# without adverse effects.
# 
# If a region of memory is freed with hipFree and a subsequent call
# to hipMalloc returns memory with the same device address,
# hipIpcGetMemHandle will return a unique handle for the
# new memory.
# 
# @param handle - Pointer to user allocated hipIpcMemHandle to return
#                    the handle in.
# @param devPtr - Base pointer to previously allocated device memory
# 
# @returns
# hipSuccess,
# hipErrorInvalidHandle,
# hipErrorOutOfMemory,
# hipErrorMapFailed,
#
cdef hipError_t hipIpcGetMemHandle(hipIpcMemHandle_st * handle,void * devPtr) nogil:
    global _hipIpcGetMemHandle__funptr
    __init_symbol(&_hipIpcGetMemHandle__funptr,"hipIpcGetMemHandle")
    return (<hipError_t (*)(hipIpcMemHandle_st *,void *) nogil> _hipIpcGetMemHandle__funptr)(handle,devPtr)


cdef void* _hipIpcOpenMemHandle__funptr = NULL
# 
# @brief Opens an interprocess memory handle exported from another process
#          and returns a device pointer usable in the local process.
# 
# Maps memory exported from another process with hipIpcGetMemHandle into
# the current device address space. For contexts on different devices
# hipIpcOpenMemHandle can attempt to enable peer access between the
# devices as if the user called hipDeviceEnablePeerAccess. This behavior is
# controlled by the hipIpcMemLazyEnablePeerAccess flag.
# hipDeviceCanAccessPeer can determine if a mapping is possible.
# 
# Contexts that may open hipIpcMemHandles are restricted in the following way.
# hipIpcMemHandles from each device in a given process may only be opened
# by one context per device per other process.
# 
# Memory returned from hipIpcOpenMemHandle must be freed with
# hipIpcCloseMemHandle.
# 
# Calling hipFree on an exported memory region before calling
# hipIpcCloseMemHandle in the importing context will result in undefined
# behavior.
# 
# @param devPtr - Returned device pointer
# @param handle - hipIpcMemHandle to open
# @param flags  - Flags for this operation. Must be specified as hipIpcMemLazyEnablePeerAccess
# 
# @returns
# hipSuccess,
# hipErrorMapFailed,
# hipErrorInvalidHandle,
# hipErrorTooManyPeers
# 
# @note During multiple processes, using the same memory handle opened by the current context,
# there is no guarantee that the same device poiter will be returned in @p *devPtr.
# This is diffrent from CUDA.
#
cdef hipError_t hipIpcOpenMemHandle(void ** devPtr,hipIpcMemHandle_st handle,unsigned int flags) nogil:
    global _hipIpcOpenMemHandle__funptr
    __init_symbol(&_hipIpcOpenMemHandle__funptr,"hipIpcOpenMemHandle")
    return (<hipError_t (*)(void **,hipIpcMemHandle_st,unsigned int) nogil> _hipIpcOpenMemHandle__funptr)(devPtr,handle,flags)


cdef void* _hipIpcCloseMemHandle__funptr = NULL
# 
# @brief Close memory mapped with hipIpcOpenMemHandle
# 
# Unmaps memory returnd by hipIpcOpenMemHandle. The original allocation
# in the exporting process as well as imported mappings in other processes
# will be unaffected.
# 
# Any resources used to enable peer access will be freed if this is the
# last mapping using them.
# 
# @param devPtr - Device pointer returned by hipIpcOpenMemHandle
# 
# @returns
# hipSuccess,
# hipErrorMapFailed,
# hipErrorInvalidHandle,
#
cdef hipError_t hipIpcCloseMemHandle(void * devPtr) nogil:
    global _hipIpcCloseMemHandle__funptr
    __init_symbol(&_hipIpcCloseMemHandle__funptr,"hipIpcCloseMemHandle")
    return (<hipError_t (*)(void *) nogil> _hipIpcCloseMemHandle__funptr)(devPtr)


cdef void* _hipIpcGetEventHandle__funptr = NULL
# 
# @brief Gets an opaque interprocess handle for an event.
# 
# This opaque handle may be copied into other processes and opened with hipIpcOpenEventHandle.
# Then hipEventRecord, hipEventSynchronize, hipStreamWaitEvent and hipEventQuery may be used in
# either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
# will result in undefined behavior.
# 
# @param[out]  handle Pointer to hipIpcEventHandle to return the opaque event handle
# @param[in]   event  Event allocated with hipEventInterprocess and hipEventDisableTiming flags
# 
# @returns #hipSuccess, #hipErrorInvalidConfiguration, #hipErrorInvalidValue
#
cdef hipError_t hipIpcGetEventHandle(hipIpcEventHandle_st * handle,hipEvent_t event) nogil:
    global _hipIpcGetEventHandle__funptr
    __init_symbol(&_hipIpcGetEventHandle__funptr,"hipIpcGetEventHandle")
    return (<hipError_t (*)(hipIpcEventHandle_st *,hipEvent_t) nogil> _hipIpcGetEventHandle__funptr)(handle,event)


cdef void* _hipIpcOpenEventHandle__funptr = NULL
# 
# @brief Opens an interprocess event handles.
# 
# Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle. The returned
# hipEvent_t behaves like a locally created event with the hipEventDisableTiming flag specified. This event
# need be freed with hipEventDestroy. Operations on the imported event after the exported event has been freed
# with hipEventDestroy will result in undefined behavior. If the function is called within the same process where
# handle is returned by hipIpcGetEventHandle, it will return hipErrorInvalidContext.
# 
# @param[out]  event  Pointer to hipEvent_t to return the event
# @param[in]   handle The opaque interprocess handle to open
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext
#
cdef hipError_t hipIpcOpenEventHandle(hipEvent_t* event,hipIpcEventHandle_st handle) nogil:
    global _hipIpcOpenEventHandle__funptr
    __init_symbol(&_hipIpcOpenEventHandle__funptr,"hipIpcOpenEventHandle")
    return (<hipError_t (*)(hipEvent_t*,hipIpcEventHandle_st) nogil> _hipIpcOpenEventHandle__funptr)(event,handle)


cdef void* _hipFuncSetAttribute__funptr = NULL
# 
# @}
# 
# 
# 
#  @defgroup Execution Execution Control
#  @{
#  This section describes the execution control functions of HIP runtime API.
# 
# 
# 
# @brief Set attribute for a specific function
# 
# @param [in] func;
# @param [in] attr;
# @param [in] value;
# 
# @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
# 
# Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
#
cdef hipError_t hipFuncSetAttribute(const void * func,hipFuncAttribute attr,int value) nogil:
    global _hipFuncSetAttribute__funptr
    __init_symbol(&_hipFuncSetAttribute__funptr,"hipFuncSetAttribute")
    return (<hipError_t (*)(const void *,hipFuncAttribute,int) nogil> _hipFuncSetAttribute__funptr)(func,attr,value)


cdef void* _hipFuncSetCacheConfig__funptr = NULL
# 
# @brief Set Cache configuration for a specific function
# 
# @param [in] config;
# 
# @returns #hipSuccess, #hipErrorNotInitialized
# Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
# on those architectures.
#
cdef hipError_t hipFuncSetCacheConfig(const void * func,hipFuncCache_t config) nogil:
    global _hipFuncSetCacheConfig__funptr
    __init_symbol(&_hipFuncSetCacheConfig__funptr,"hipFuncSetCacheConfig")
    return (<hipError_t (*)(const void *,hipFuncCache_t) nogil> _hipFuncSetCacheConfig__funptr)(func,config)


cdef void* _hipFuncSetSharedMemConfig__funptr = NULL
# 
# @brief Set shared memory configuation for a specific function
# 
# @param [in] func
# @param [in] config
# 
# @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
# 
# Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
#
cdef hipError_t hipFuncSetSharedMemConfig(const void * func,hipSharedMemConfig config) nogil:
    global _hipFuncSetSharedMemConfig__funptr
    __init_symbol(&_hipFuncSetSharedMemConfig__funptr,"hipFuncSetSharedMemConfig")
    return (<hipError_t (*)(const void *,hipSharedMemConfig) nogil> _hipFuncSetSharedMemConfig__funptr)(func,config)


cdef void* _hipGetLastError__funptr = NULL
# 
#  @}
# 
# 
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#   @defgroup Error Error Handling
#   @{
#   This section describes the error handling functions of HIP runtime API.
# 
# 
#  @brief Return last error returned by any HIP runtime API call and resets the stored error code to
#  #hipSuccess
# 
#  @returns return code from last HIP called from the active host thread
# 
#  Returns the last error that has been returned by any of the runtime calls in the same host
#  thread, and then resets the saved error to #hipSuccess.
# 
#  @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
cdef hipError_t hipGetLastError() nogil:
    global _hipGetLastError__funptr
    __init_symbol(&_hipGetLastError__funptr,"hipGetLastError")
    return (<hipError_t (*)() nogil> _hipGetLastError__funptr)()


cdef void* _hipPeekAtLastError__funptr = NULL
# 
# @brief Return last error returned by any HIP runtime API call.
# 
# @return #hipSuccess
# 
# Returns the last error that has been returned by any of the runtime calls in the same host
# thread. Unlike hipGetLastError, this function does not reset the saved error code.
# 
# @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
cdef hipError_t hipPeekAtLastError() nogil:
    global _hipPeekAtLastError__funptr
    __init_symbol(&_hipPeekAtLastError__funptr,"hipPeekAtLastError")
    return (<hipError_t (*)() nogil> _hipPeekAtLastError__funptr)()


cdef void* _hipGetErrorName__funptr = NULL
# 
# @brief Return hip error as text string form.
# 
# @param hip_error Error code to convert to name.
# @return const char pointer to the NULL-terminated error name
# 
# @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
cdef const char * hipGetErrorName(hipError_t hip_error) nogil:
    global _hipGetErrorName__funptr
    __init_symbol(&_hipGetErrorName__funptr,"hipGetErrorName")
    return (<const char * (*)(hipError_t) nogil> _hipGetErrorName__funptr)(hip_error)


cdef void* _hipGetErrorString__funptr = NULL
# 
# @brief Return handy text string message to explain the error which occurred
# 
# @param hipError Error code to convert to string.
# @return const char pointer to the NULL-terminated error string
# 
# @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
cdef const char * hipGetErrorString(hipError_t hipError) nogil:
    global _hipGetErrorString__funptr
    __init_symbol(&_hipGetErrorString__funptr,"hipGetErrorString")
    return (<const char * (*)(hipError_t) nogil> _hipGetErrorString__funptr)(hipError)


cdef void* _hipDrvGetErrorName__funptr = NULL
# 
# @brief Return hip error as text string form.
# 
# @param [in] hipError Error code to convert to string.
# @param [out] const char pointer to the NULL-terminated error string
# @return #hipSuccess, #hipErrorInvalidValue
# 
# @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
cdef hipError_t hipDrvGetErrorName(hipError_t hipError,const char ** errorString) nogil:
    global _hipDrvGetErrorName__funptr
    __init_symbol(&_hipDrvGetErrorName__funptr,"hipDrvGetErrorName")
    return (<hipError_t (*)(hipError_t,const char **) nogil> _hipDrvGetErrorName__funptr)(hipError,errorString)


cdef void* _hipDrvGetErrorString__funptr = NULL
# 
# @brief Return handy text string message to explain the error which occurred
# 
# @param [in] hipError Error code to convert to string.
# @param [out] const char pointer to the NULL-terminated error string
# @return #hipSuccess, #hipErrorInvalidValue
# 
# @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
cdef hipError_t hipDrvGetErrorString(hipError_t hipError,const char ** errorString) nogil:
    global _hipDrvGetErrorString__funptr
    __init_symbol(&_hipDrvGetErrorString__funptr,"hipDrvGetErrorString")
    return (<hipError_t (*)(hipError_t,const char **) nogil> _hipDrvGetErrorString__funptr)(hipError,errorString)


cdef void* _hipStreamCreate__funptr = NULL
# 
# @brief Create an asynchronous stream.
# 
# @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
# newly created stream.
# @return #hipSuccess, #hipErrorInvalidValue
# 
# Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
# reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
# the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
# used by the stream, applicaiton must call hipStreamDestroy.
# 
# @return #hipSuccess, #hipErrorInvalidValue
# 
# @see hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipStreamCreate(hipStream_t* stream) nogil:
    global _hipStreamCreate__funptr
    __init_symbol(&_hipStreamCreate__funptr,"hipStreamCreate")
    return (<hipError_t (*)(hipStream_t*) nogil> _hipStreamCreate__funptr)(stream)


cdef void* _hipStreamCreateWithFlags__funptr = NULL
# 
# @brief Create an asynchronous stream.
# 
# @param[in, out] stream Pointer to new stream
# @param[in ] flags to control stream creation.
# @return #hipSuccess, #hipErrorInvalidValue
# 
# Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
# reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
# the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
# used by the stream, applicaiton must call hipStreamDestroy. Flags controls behavior of the
# stream.  See #hipStreamDefault, #hipStreamNonBlocking.
# 
# 
# @see hipStreamCreate, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipStreamCreateWithFlags(hipStream_t* stream,unsigned int flags) nogil:
    global _hipStreamCreateWithFlags__funptr
    __init_symbol(&_hipStreamCreateWithFlags__funptr,"hipStreamCreateWithFlags")
    return (<hipError_t (*)(hipStream_t*,unsigned int) nogil> _hipStreamCreateWithFlags__funptr)(stream,flags)


cdef void* _hipStreamCreateWithPriority__funptr = NULL
# 
# @brief Create an asynchronous stream with the specified priority.
# 
# @param[in, out] stream Pointer to new stream
# @param[in ] flags to control stream creation.
# @param[in ] priority of the stream. Lower numbers represent higher priorities.
# @return #hipSuccess, #hipErrorInvalidValue
# 
# Create a new asynchronous stream with the specified priority.  @p stream returns an opaque handle
# that can be used to reference the newly created stream in subsequent hipStream* commands.  The
# stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
# To release the memory used by the stream, applicaiton must call hipStreamDestroy. Flags controls
# behavior of the stream.  See #hipStreamDefault, #hipStreamNonBlocking.
# 
# 
# @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipStreamCreateWithPriority(hipStream_t* stream,unsigned int flags,int priority) nogil:
    global _hipStreamCreateWithPriority__funptr
    __init_symbol(&_hipStreamCreateWithPriority__funptr,"hipStreamCreateWithPriority")
    return (<hipError_t (*)(hipStream_t*,unsigned int,int) nogil> _hipStreamCreateWithPriority__funptr)(stream,flags,priority)


cdef void* _hipDeviceGetStreamPriorityRange__funptr = NULL
# 
# @brief Returns numerical values that correspond to the least and greatest stream priority.
# 
# @param[in, out] leastPriority pointer in which value corresponding to least priority is returned.
# @param[in, out] greatestPriority pointer in which value corresponding to greatest priority is returned.
# 
# Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least
# and greatest stream priority respectively. Stream priorities follow a convention where lower numbers
# imply greater priorities. The range of meaningful stream priorities is given by
# [*greatestPriority, *leastPriority]. If the user attempts to create a stream with a priority value
# that is outside the the meaningful range as specified by this API, the priority is automatically
# clamped to within the valid range.
cdef hipError_t hipDeviceGetStreamPriorityRange(int * leastPriority,int * greatestPriority) nogil:
    global _hipDeviceGetStreamPriorityRange__funptr
    __init_symbol(&_hipDeviceGetStreamPriorityRange__funptr,"hipDeviceGetStreamPriorityRange")
    return (<hipError_t (*)(int *,int *) nogil> _hipDeviceGetStreamPriorityRange__funptr)(leastPriority,greatestPriority)


cdef void* _hipStreamDestroy__funptr = NULL
# 
# @brief Destroys the specified stream.
# 
# @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
# newly created stream.
# @return #hipSuccess #hipErrorInvalidHandle
# 
# Destroys the specified stream.
# 
# If commands are still executing on the specified stream, some may complete execution before the
# queue is deleted.
# 
# The queue may be destroyed while some commands are still inflight, or may wait for all commands
# queued to the stream before destroying it.
# 
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamQuery, hipStreamWaitEvent,
# hipStreamSynchronize
cdef hipError_t hipStreamDestroy(hipStream_t stream) nogil:
    global _hipStreamDestroy__funptr
    __init_symbol(&_hipStreamDestroy__funptr,"hipStreamDestroy")
    return (<hipError_t (*)(hipStream_t) nogil> _hipStreamDestroy__funptr)(stream)


cdef void* _hipStreamQuery__funptr = NULL
# 
# @brief Return #hipSuccess if all of the operations in the specified @p stream have completed, or
# #hipErrorNotReady if not.
# 
# @param[in] stream stream to query
# 
# @return #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle
# 
# This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
# host threads are sending work to the stream, the status may change immediately after the function
# is called.  It is typically used for debug.
# 
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamSynchronize,
# hipStreamDestroy
cdef hipError_t hipStreamQuery(hipStream_t stream) nogil:
    global _hipStreamQuery__funptr
    __init_symbol(&_hipStreamQuery__funptr,"hipStreamQuery")
    return (<hipError_t (*)(hipStream_t) nogil> _hipStreamQuery__funptr)(stream)


cdef void* _hipStreamSynchronize__funptr = NULL
# 
# @brief Wait for all commands in stream to complete.
# 
# @param[in] stream stream identifier.
# 
# @return #hipSuccess, #hipErrorInvalidHandle
# 
# This command is host-synchronous : the host will block until the specified stream is empty.
# 
# This command follows standard null-stream semantics.  Specifically, specifying the null stream
# will cause the command to wait for other streams on the same device to complete all pending
# operations.
# 
# This command honors the hipDeviceLaunchBlocking flag, which controls whether the wait is active
# or blocking.
# 
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamDestroy
#
cdef hipError_t hipStreamSynchronize(hipStream_t stream) nogil:
    global _hipStreamSynchronize__funptr
    __init_symbol(&_hipStreamSynchronize__funptr,"hipStreamSynchronize")
    return (<hipError_t (*)(hipStream_t) nogil> _hipStreamSynchronize__funptr)(stream)


cdef void* _hipStreamWaitEvent__funptr = NULL
# 
# @brief Make the specified compute stream wait for an event
# 
# @param[in] stream stream to make wait.
# @param[in] event event to wait on
# @param[in] flags control operation [must be 0]
# 
# @return #hipSuccess, #hipErrorInvalidHandle
# 
# This function inserts a wait operation into the specified stream.
# All future work submitted to @p stream will wait until @p event reports completion before
# beginning execution.
# 
# This function only waits for commands in the current stream to complete.  Notably,, this function
# does not impliciy wait for commands in the default stream to complete, even if the specified
# stream is created with hipStreamNonBlocking = 0.
# 
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamDestroy
cdef hipError_t hipStreamWaitEvent(hipStream_t stream,hipEvent_t event,unsigned int flags) nogil:
    global _hipStreamWaitEvent__funptr
    __init_symbol(&_hipStreamWaitEvent__funptr,"hipStreamWaitEvent")
    return (<hipError_t (*)(hipStream_t,hipEvent_t,unsigned int) nogil> _hipStreamWaitEvent__funptr)(stream,event,flags)


cdef void* _hipStreamGetFlags__funptr = NULL
# 
# @brief Return flags associated with this stream.
# 
# @param[in] stream stream to be queried
# @param[in,out] flags Pointer to an unsigned integer in which the stream's flags are returned
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
# 
# @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
# 
# Return flags associated with this stream in *@p flags.
# 
# @see hipStreamCreateWithFlags
cdef hipError_t hipStreamGetFlags(hipStream_t stream,unsigned int * flags) nogil:
    global _hipStreamGetFlags__funptr
    __init_symbol(&_hipStreamGetFlags__funptr,"hipStreamGetFlags")
    return (<hipError_t (*)(hipStream_t,unsigned int *) nogil> _hipStreamGetFlags__funptr)(stream,flags)


cdef void* _hipStreamGetPriority__funptr = NULL
# 
# @brief Query the priority of a stream.
# 
# @param[in] stream stream to be queried
# @param[in,out] priority Pointer to an unsigned integer in which the stream's priority is returned
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
# 
# @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
# 
# Query the priority of a stream. The priority is returned in in priority.
# 
# @see hipStreamCreateWithFlags
cdef hipError_t hipStreamGetPriority(hipStream_t stream,int * priority) nogil:
    global _hipStreamGetPriority__funptr
    __init_symbol(&_hipStreamGetPriority__funptr,"hipStreamGetPriority")
    return (<hipError_t (*)(hipStream_t,int *) nogil> _hipStreamGetPriority__funptr)(stream,priority)


cdef void* _hipExtStreamCreateWithCUMask__funptr = NULL
# 
# @brief Create an asynchronous stream with the specified CU mask.
# 
# @param[in, out] stream Pointer to new stream
# @param[in ] cuMaskSize Size of CU mask bit array passed in.
# @param[in ] cuMask Bit-vector representing the CU mask. Each active bit represents using one CU.
# The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
# CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are ignored.
# It is user's responsibility to make sure the input is meaningful.
# @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
# 
# Create a new asynchronous stream with the specified CU mask.  @p stream returns an opaque handle
# that can be used to reference the newly created stream in subsequent hipStream* commands.  The
# stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
# To release the memory used by the stream, application must call hipStreamDestroy.
# 
# 
# @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream,unsigned int cuMaskSize,const unsigned int * cuMask) nogil:
    global _hipExtStreamCreateWithCUMask__funptr
    __init_symbol(&_hipExtStreamCreateWithCUMask__funptr,"hipExtStreamCreateWithCUMask")
    return (<hipError_t (*)(hipStream_t*,unsigned int,const unsigned int *) nogil> _hipExtStreamCreateWithCUMask__funptr)(stream,cuMaskSize,cuMask)


cdef void* _hipExtStreamGetCUMask__funptr = NULL
# 
# @brief Get CU mask associated with an asynchronous stream
# 
# @param[in] stream stream to be queried
# @param[in] cuMaskSize number of the block of memories (uint32_t *) allocated by user
# @param[out] cuMask Pointer to a pre-allocated block of memories (uint32_t *) in which
# the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
# each active bit represents one active CU
# @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
# 
# @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
cdef hipError_t hipExtStreamGetCUMask(hipStream_t stream,unsigned int cuMaskSize,unsigned int * cuMask) nogil:
    global _hipExtStreamGetCUMask__funptr
    __init_symbol(&_hipExtStreamGetCUMask__funptr,"hipExtStreamGetCUMask")
    return (<hipError_t (*)(hipStream_t,unsigned int,unsigned int *) nogil> _hipExtStreamGetCUMask__funptr)(stream,cuMaskSize,cuMask)


cdef void* _hipStreamAddCallback__funptr = NULL
# 
# @brief Adds a callback to be called on the host after all currently enqueued
# items in the stream have completed.  For each
# hipStreamAddCallback call, a callback will be executed exactly once.
# The callback will block later work in the stream until it is finished.
# @param[in] stream   - Stream to add callback to
# @param[in] callback - The function to call once preceding stream operations are complete
# @param[in] userData - User specified data to be passed to the callback function
# @param[in] flags    - Reserved for future use, must be 0
# @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorNotSupported
# 
# @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamSynchronize,
# hipStreamWaitEvent, hipStreamDestroy, hipStreamCreateWithPriority
#
cdef hipError_t hipStreamAddCallback(hipStream_t stream,hipStreamCallback_t callback,void * userData,unsigned int flags):
    global _hipStreamAddCallback__funptr
    __init_symbol(&_hipStreamAddCallback__funptr,"hipStreamAddCallback")
    return (<hipError_t (*)(hipStream_t,hipStreamCallback_t,void *,unsigned int)> _hipStreamAddCallback__funptr)(stream,callback,userData,flags)


cdef void* _hipStreamWaitValue32__funptr = NULL
# 
#  @}
# 
# 
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#   @defgroup StreamM Stream Memory Operations
#   @{
#   This section describes Stream Memory Wait and Write functions of HIP runtime API.
# 
# 
#  @brief Enqueues a wait command to the stream.[BETA]
# 
#  @param [in] stream - Stream identifier
#  @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
#  @param [in] value  - Value to be used in compare operation
#  @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
#  hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor
#  @param [in] mask   - Mask to be applied on value at memory before it is compared with value,
#  default value is set to enable every bit
# 
#  @returns #hipSuccess, #hipErrorInvalidValue
# 
#  Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
#  not execute until the defined wait condition is true.
# 
#  hipStreamWaitValueGte: waits until *ptr&mask >= value
#  hipStreamWaitValueEq : waits until *ptr&mask == value
#  hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
#  hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
# 
#  @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
# 
#  @note Support for hipStreamWaitValue32 can be queried using 'hipDeviceGetAttribute()' and
#  'hipDeviceAttributeCanUseStreamWaitValue' flag.
# 
#  @beta This API is marked as beta, meaning, while this is feature complete,
#  it is still open to changes and may have outstanding issues.
# 
#  @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue64, hipStreamWriteValue64,
#  hipStreamWriteValue32, hipDeviceGetAttribute
cdef hipError_t hipStreamWaitValue32(hipStream_t stream,void * ptr,unsigned int value,unsigned int flags,unsigned int mask) nogil:
    global _hipStreamWaitValue32__funptr
    __init_symbol(&_hipStreamWaitValue32__funptr,"hipStreamWaitValue32")
    return (<hipError_t (*)(hipStream_t,void *,unsigned int,unsigned int,unsigned int) nogil> _hipStreamWaitValue32__funptr)(stream,ptr,value,flags,mask)


cdef void* _hipStreamWaitValue64__funptr = NULL
# 
# @brief Enqueues a wait command to the stream.[BETA]
# 
# @param [in] stream - Stream identifier
# @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
# @param [in] value  - Value to be used in compare operation
# @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
# hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor.
# @param [in] mask   - Mask to be applied on value at memory before it is compared with value
# default value is set to enable every bit
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
# not execute until the defined wait condition is true.
# 
# hipStreamWaitValueGte: waits until *ptr&mask >= value
# hipStreamWaitValueEq : waits until *ptr&mask == value
# hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
# hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
# 
# @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
# 
# @note Support for hipStreamWaitValue64 can be queried using 'hipDeviceGetAttribute()' and
# 'hipDeviceAttributeCanUseStreamWaitValue' flag.
# 
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# 
# @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue32, hipStreamWriteValue64,
# hipStreamWriteValue32, hipDeviceGetAttribute
cdef hipError_t hipStreamWaitValue64(hipStream_t stream,void * ptr,unsigned long value,unsigned int flags,unsigned long mask) nogil:
    global _hipStreamWaitValue64__funptr
    __init_symbol(&_hipStreamWaitValue64__funptr,"hipStreamWaitValue64")
    return (<hipError_t (*)(hipStream_t,void *,unsigned long,unsigned int,unsigned long) nogil> _hipStreamWaitValue64__funptr)(stream,ptr,value,flags,mask)


cdef void* _hipStreamWriteValue32__funptr = NULL
# 
# @brief Enqueues a write command to the stream.[BETA]
# 
# @param [in] stream - Stream identifier
# @param [in] ptr    - Pointer to a GPU accessible memory object
# @param [in] value  - Value to be written
# @param [in] flags  - reserved, ignored for now, will be used in future releases
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# Enqueues a write command to the stream, write operation is performed after all earlier commands
# on this stream have completed the execution.
# 
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# 
# @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
# hipStreamWaitValue64
cdef hipError_t hipStreamWriteValue32(hipStream_t stream,void * ptr,unsigned int value,unsigned int flags) nogil:
    global _hipStreamWriteValue32__funptr
    __init_symbol(&_hipStreamWriteValue32__funptr,"hipStreamWriteValue32")
    return (<hipError_t (*)(hipStream_t,void *,unsigned int,unsigned int) nogil> _hipStreamWriteValue32__funptr)(stream,ptr,value,flags)


cdef void* _hipStreamWriteValue64__funptr = NULL
# 
# @brief Enqueues a write command to the stream.[BETA]
# 
# @param [in] stream - Stream identifier
# @param [in] ptr    - Pointer to a GPU accessible memory object
# @param [in] value  - Value to be written
# @param [in] flags  - reserved, ignored for now, will be used in future releases
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# Enqueues a write command to the stream, write operation is performed after all earlier commands
# on this stream have completed the execution.
# 
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# 
# @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
# hipStreamWaitValue64
cdef hipError_t hipStreamWriteValue64(hipStream_t stream,void * ptr,unsigned long value,unsigned int flags) nogil:
    global _hipStreamWriteValue64__funptr
    __init_symbol(&_hipStreamWriteValue64__funptr,"hipStreamWriteValue64")
    return (<hipError_t (*)(hipStream_t,void *,unsigned long,unsigned int) nogil> _hipStreamWriteValue64__funptr)(stream,ptr,value,flags)


cdef void* _hipEventCreateWithFlags__funptr = NULL
# 
#   @}
# 
# 
#  -------------------------------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------------------
#    @defgroup Event Event Management
#    @{
#    This section describes the event management functions of HIP runtime API.
# 
# 
#   @brief Create an event with the specified flags
# 
#   @param[in,out] event Returns the newly created event.
#   @param[in] flags     Flags to control event behavior.  Valid values are #hipEventDefault,
# #hipEventBlockingSync, #hipEventDisableTiming, #hipEventInterprocess
#   #hipEventDefault : Default flag.  The event will use active synchronization and will support
# timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a
# CPU to poll on the event.
#   #hipEventBlockingSync : The event will use blocking synchronization : if hipEventSynchronize is
# called on this event, the thread will block until the event completes.  This can increase latency
# for the synchroniation but can result in lower power and more resources for other CPU threads.
#   #hipEventDisableTiming : Disable recording of timing information. Events created with this flag
# would not record profiling data and provide best performance if used for synchronization.
#   #hipEventInterprocess : The event can be used as an interprocess event. hipEventDisableTiming
# flag also must be set when hipEventInterprocess flag is set.
# 
#   @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
# #hipErrorLaunchFailure, #hipErrorOutOfMemory
# 
#   @see hipEventCreate, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
cdef hipError_t hipEventCreateWithFlags(hipEvent_t* event,unsigned int flags) nogil:
    global _hipEventCreateWithFlags__funptr
    __init_symbol(&_hipEventCreateWithFlags__funptr,"hipEventCreateWithFlags")
    return (<hipError_t (*)(hipEvent_t*,unsigned int) nogil> _hipEventCreateWithFlags__funptr)(event,flags)


cdef void* _hipEventCreate__funptr = NULL
# 
#  Create an event
# 
# @param[in,out] event Returns the newly created event.
# 
# @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
# #hipErrorLaunchFailure, #hipErrorOutOfMemory
# 
# @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize,
# hipEventDestroy, hipEventElapsedTime
cdef hipError_t hipEventCreate(hipEvent_t* event) nogil:
    global _hipEventCreate__funptr
    __init_symbol(&_hipEventCreate__funptr,"hipEventCreate")
    return (<hipError_t (*)(hipEvent_t*) nogil> _hipEventCreate__funptr)(event)


cdef void* _hipEventRecord__funptr = NULL
cdef hipError_t hipEventRecord(hipEvent_t event,hipStream_t stream) nogil:
    global _hipEventRecord__funptr
    __init_symbol(&_hipEventRecord__funptr,"hipEventRecord")
    return (<hipError_t (*)(hipEvent_t,hipStream_t) nogil> _hipEventRecord__funptr)(event,stream)


cdef void* _hipEventDestroy__funptr = NULL
# 
#  @brief Destroy the specified event.
# 
#  @param[in] event Event to destroy.
#  @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
# #hipErrorLaunchFailure
# 
#  Releases memory associated with the event.  If the event is recording but has not completed
# recording when hipEventDestroy() is called, the function will return immediately and the
# completion_future resources will be released later, when the hipDevice is synchronized.
# 
# @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize, hipEventRecord,
# hipEventElapsedTime
# 
# @returns #hipSuccess
cdef hipError_t hipEventDestroy(hipEvent_t event) nogil:
    global _hipEventDestroy__funptr
    __init_symbol(&_hipEventDestroy__funptr,"hipEventDestroy")
    return (<hipError_t (*)(hipEvent_t) nogil> _hipEventDestroy__funptr)(event)


cdef void* _hipEventSynchronize__funptr = NULL
# 
#  @brief Wait for an event to complete.
# 
#  This function will block until the event is ready, waiting for all previous work in the stream
# specified when event was recorded with hipEventRecord().
# 
#  If hipEventRecord() has not been called on @p event, this function returns immediately.
# 
#  TODO-hip- This function needs to support hipEventBlockingSync parameter.
# 
#  @param[in] event Event on which to wait.
#  @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized,
# #hipErrorInvalidHandle, #hipErrorLaunchFailure
# 
#  @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
# hipEventElapsedTime
cdef hipError_t hipEventSynchronize(hipEvent_t event) nogil:
    global _hipEventSynchronize__funptr
    __init_symbol(&_hipEventSynchronize__funptr,"hipEventSynchronize")
    return (<hipError_t (*)(hipEvent_t) nogil> _hipEventSynchronize__funptr)(event)


cdef void* _hipEventElapsedTime__funptr = NULL
# 
# @brief Return the elapsed time between two events.
# 
# @param[out] ms : Return time between start and stop in ms.
# @param[in]   start : Start event.
# @param[in]   stop  : Stop event.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotReady, #hipErrorInvalidHandle,
# #hipErrorNotInitialized, #hipErrorLaunchFailure
# 
# Computes the elapsed time between two events. Time is computed in ms, with
# a resolution of approximately 1 us.
# 
# Events which are recorded in a NULL stream will block until all commands
# on all other streams complete execution, and then record the timestamp.
# 
# Events which are recorded in a non-NULL stream will record their timestamp
# when they reach the head of the specified stream, after all previous
# commands in that stream have completed executing.  Thus the time that
# the event recorded may be significantly after the host calls hipEventRecord().
# 
# If hipEventRecord() has not been called on either event, then #hipErrorInvalidHandle is
# returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
# recorded on one or both events (that is, hipEventQuery() would return #hipErrorNotReady on at
# least one of the events), then #hipErrorNotReady is returned.
# 
# @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
# hipEventSynchronize
cdef hipError_t hipEventElapsedTime(float * ms,hipEvent_t start,hipEvent_t stop) nogil:
    global _hipEventElapsedTime__funptr
    __init_symbol(&_hipEventElapsedTime__funptr,"hipEventElapsedTime")
    return (<hipError_t (*)(float *,hipEvent_t,hipEvent_t) nogil> _hipEventElapsedTime__funptr)(ms,start,stop)


cdef void* _hipEventQuery__funptr = NULL
# 
# @brief Query event status
# 
# @param[in] event Event to query.
# @returns #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle, #hipErrorInvalidValue,
# #hipErrorNotInitialized, #hipErrorLaunchFailure
# 
# Query the status of the specified event.  This function will return #hipSuccess if all
# commands in the appropriate stream (specified to hipEventRecord()) have completed.  If that work
# has not completed, or if hipEventRecord() was not called on the event, then #hipErrorNotReady is
# returned.
# 
# @see hipEventCreate, hipEventCreateWithFlags, hipEventRecord, hipEventDestroy,
# hipEventSynchronize, hipEventElapsedTime
cdef hipError_t hipEventQuery(hipEvent_t event) nogil:
    global _hipEventQuery__funptr
    __init_symbol(&_hipEventQuery__funptr,"hipEventQuery")
    return (<hipError_t (*)(hipEvent_t) nogil> _hipEventQuery__funptr)(event)


cdef void* _hipPointerGetAttributes__funptr = NULL
# 
#  @}
# 
# 
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#   @defgroup Memory Memory Management
#   @{
#   This section describes the memory management functions of HIP runtime API.
#   The following CUDA APIs are not currently supported:
#   - cudaMalloc3D
#   - cudaMalloc3DArray
#   - TODO - more 2D, 3D, array APIs here.
# 
# 
# 
# 
#   @brief Return attributes for the specified pointer
# 
#   @param [out]  attributes  attributes for the specified pointer
#   @param [in]   ptr         pointer to get attributes for
# 
#   @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
#   @see hipPointerGetAttribute
cdef hipError_t hipPointerGetAttributes(hipPointerAttribute_t * attributes,const void * ptr) nogil:
    global _hipPointerGetAttributes__funptr
    __init_symbol(&_hipPointerGetAttributes__funptr,"hipPointerGetAttributes")
    return (<hipError_t (*)(hipPointerAttribute_t *,const void *) nogil> _hipPointerGetAttributes__funptr)(attributes,ptr)


cdef void* _hipPointerGetAttribute__funptr = NULL
# 
# @brief Returns information about the specified pointer.[BETA]
# 
# @param [in, out] data     returned pointer attribute value
# @param [in]      atribute attribute to query for
# @param [in]      ptr      pointer to get attributes for
# 
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# 
# @see hipPointerGetAttributes
cdef hipError_t hipPointerGetAttribute(void * data,hipPointer_attribute attribute,void * ptr) nogil:
    global _hipPointerGetAttribute__funptr
    __init_symbol(&_hipPointerGetAttribute__funptr,"hipPointerGetAttribute")
    return (<hipError_t (*)(void *,hipPointer_attribute,void *) nogil> _hipPointerGetAttribute__funptr)(data,attribute,ptr)


cdef void* _hipDrvPointerGetAttributes__funptr = NULL
# 
# @brief Returns information about the specified pointer.[BETA]
# 
# @param [in]  numAttributes   number of attributes to query for
# @param [in]  attributes      attributes to query for
# @param [in, out] data        a two-dimensional containing pointers to memory locations
#                              where the result of each attribute query will be written to
# @param [in]  ptr             pointer to get attributes for
# 
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @beta This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
# 
# @see hipPointerGetAttribute
cdef hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes,hipPointer_attribute * attributes,void ** data,void * ptr) nogil:
    global _hipDrvPointerGetAttributes__funptr
    __init_symbol(&_hipDrvPointerGetAttributes__funptr,"hipDrvPointerGetAttributes")
    return (<hipError_t (*)(unsigned int,hipPointer_attribute *,void **,void *) nogil> _hipDrvPointerGetAttributes__funptr)(numAttributes,attributes,data,ptr)


cdef void* _hipImportExternalSemaphore__funptr = NULL
# 
# @brief Imports an external semaphore.
# 
# @param[out] extSem_out  External semaphores to be waited on
# @param[in] semHandleDesc Semaphore import handle descriptor
# 
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @see
cdef hipError_t hipImportExternalSemaphore(void ** extSem_out,hipExternalSemaphoreHandleDesc_st * semHandleDesc) nogil:
    global _hipImportExternalSemaphore__funptr
    __init_symbol(&_hipImportExternalSemaphore__funptr,"hipImportExternalSemaphore")
    return (<hipError_t (*)(void **,hipExternalSemaphoreHandleDesc_st *) nogil> _hipImportExternalSemaphore__funptr)(extSem_out,semHandleDesc)


cdef void* _hipSignalExternalSemaphoresAsync__funptr = NULL
# 
# @brief Signals a set of external semaphore objects.
# 
# @param[in] extSem_out  External semaphores to be waited on
# @param[in] paramsArray Array of semaphore parameters
# @param[in] numExtSems Number of semaphores to wait on
# @param[in] stream Stream to enqueue the wait operations in
# 
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @see
cdef hipError_t hipSignalExternalSemaphoresAsync(void *const * extSemArray,hipExternalSemaphoreSignalParams_st * paramsArray,unsigned int numExtSems,hipStream_t stream) nogil:
    global _hipSignalExternalSemaphoresAsync__funptr
    __init_symbol(&_hipSignalExternalSemaphoresAsync__funptr,"hipSignalExternalSemaphoresAsync")
    return (<hipError_t (*)(void *const *,hipExternalSemaphoreSignalParams_st *,unsigned int,hipStream_t) nogil> _hipSignalExternalSemaphoresAsync__funptr)(extSemArray,paramsArray,numExtSems,stream)


cdef void* _hipWaitExternalSemaphoresAsync__funptr = NULL
# 
# @brief Waits on a set of external semaphore objects
# 
# @param[in] extSem_out  External semaphores to be waited on
# @param[in] paramsArray Array of semaphore parameters
# @param[in] numExtSems Number of semaphores to wait on
# @param[in] stream Stream to enqueue the wait operations in
# 
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @see
cdef hipError_t hipWaitExternalSemaphoresAsync(void *const * extSemArray,hipExternalSemaphoreWaitParams_st * paramsArray,unsigned int numExtSems,hipStream_t stream) nogil:
    global _hipWaitExternalSemaphoresAsync__funptr
    __init_symbol(&_hipWaitExternalSemaphoresAsync__funptr,"hipWaitExternalSemaphoresAsync")
    return (<hipError_t (*)(void *const *,hipExternalSemaphoreWaitParams_st *,unsigned int,hipStream_t) nogil> _hipWaitExternalSemaphoresAsync__funptr)(extSemArray,paramsArray,numExtSems,stream)


cdef void* _hipDestroyExternalSemaphore__funptr = NULL
# 
# @brief Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.
# 
# @param[in] extSem handle to an external memory object
# 
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @see
cdef hipError_t hipDestroyExternalSemaphore(void * extSem) nogil:
    global _hipDestroyExternalSemaphore__funptr
    __init_symbol(&_hipDestroyExternalSemaphore__funptr,"hipDestroyExternalSemaphore")
    return (<hipError_t (*)(void *) nogil> _hipDestroyExternalSemaphore__funptr)(extSem)


cdef void* _hipImportExternalMemory__funptr = NULL
# 
# @brief Imports an external memory object.
# 
# @param[out] extMem_out  Returned handle to an external memory object
# @param[in]  memHandleDesc Memory import handle descriptor
# 
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @see
cdef hipError_t hipImportExternalMemory(void ** extMem_out,hipExternalMemoryHandleDesc_st * memHandleDesc) nogil:
    global _hipImportExternalMemory__funptr
    __init_symbol(&_hipImportExternalMemory__funptr,"hipImportExternalMemory")
    return (<hipError_t (*)(void **,hipExternalMemoryHandleDesc_st *) nogil> _hipImportExternalMemory__funptr)(extMem_out,memHandleDesc)


cdef void* _hipExternalMemoryGetMappedBuffer__funptr = NULL
# 
# @brief Maps a buffer onto an imported memory object.
# 
# @param[out] devPtr Returned device pointer to buffer
# @param[in]  extMem  Handle to external memory object
# @param[in]  bufferDesc  Buffer descriptor
# 
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @see
cdef hipError_t hipExternalMemoryGetMappedBuffer(void ** devPtr,void * extMem,hipExternalMemoryBufferDesc_st * bufferDesc) nogil:
    global _hipExternalMemoryGetMappedBuffer__funptr
    __init_symbol(&_hipExternalMemoryGetMappedBuffer__funptr,"hipExternalMemoryGetMappedBuffer")
    return (<hipError_t (*)(void **,void *,hipExternalMemoryBufferDesc_st *) nogil> _hipExternalMemoryGetMappedBuffer__funptr)(devPtr,extMem,bufferDesc)


cdef void* _hipDestroyExternalMemory__funptr = NULL
# 
# @brief Destroys an external memory object.
# 
# @param[in] extMem  External memory object to be destroyed
# 
# @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
# 
# @see
cdef hipError_t hipDestroyExternalMemory(void * extMem) nogil:
    global _hipDestroyExternalMemory__funptr
    __init_symbol(&_hipDestroyExternalMemory__funptr,"hipDestroyExternalMemory")
    return (<hipError_t (*)(void *) nogil> _hipDestroyExternalMemory__funptr)(extMem)


cdef void* _hipMalloc__funptr = NULL
# 
#  @brief Allocate memory on the default accelerator
# 
#  @param[out] ptr Pointer to the allocated memory
#  @param[in]  size Requested memory size
# 
#  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# 
#  @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
# 
#  @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
# hipHostFree, hipHostMalloc
cdef hipError_t hipMalloc(void ** ptr,unsigned long size) nogil:
    global _hipMalloc__funptr
    __init_symbol(&_hipMalloc__funptr,"hipMalloc")
    return (<hipError_t (*)(void **,unsigned long) nogil> _hipMalloc__funptr)(ptr,size)


cdef void* _hipExtMallocWithFlags__funptr = NULL
# 
#  @brief Allocate memory on the default accelerator
# 
#  @param[out] ptr Pointer to the allocated memory
#  @param[in]  size Requested memory size
#  @param[in]  flags Type of memory allocation
# 
#  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# 
#  @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
# 
#  @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
# hipHostFree, hipHostMalloc
cdef hipError_t hipExtMallocWithFlags(void ** ptr,unsigned long sizeBytes,unsigned int flags) nogil:
    global _hipExtMallocWithFlags__funptr
    __init_symbol(&_hipExtMallocWithFlags__funptr,"hipExtMallocWithFlags")
    return (<hipError_t (*)(void **,unsigned long,unsigned int) nogil> _hipExtMallocWithFlags__funptr)(ptr,sizeBytes,flags)


cdef void* _hipMallocHost__funptr = NULL
# 
# @brief Allocate pinned host memory [Deprecated]
# 
# @param[out] ptr Pointer to the allocated host pinned memory
# @param[in]  size Requested memory size
# 
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# 
# @return #hipSuccess, #hipErrorOutOfMemory
# 
# @deprecated use hipHostMalloc() instead
cdef hipError_t hipMallocHost(void ** ptr,unsigned long size) nogil:
    global _hipMallocHost__funptr
    __init_symbol(&_hipMallocHost__funptr,"hipMallocHost")
    return (<hipError_t (*)(void **,unsigned long) nogil> _hipMallocHost__funptr)(ptr,size)


cdef void* _hipMemAllocHost__funptr = NULL
# 
# @brief Allocate pinned host memory [Deprecated]
# 
# @param[out] ptr Pointer to the allocated host pinned memory
# @param[in]  size Requested memory size
# 
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# 
# @return #hipSuccess, #hipErrorOutOfMemory
# 
# @deprecated use hipHostMalloc() instead
cdef hipError_t hipMemAllocHost(void ** ptr,unsigned long size) nogil:
    global _hipMemAllocHost__funptr
    __init_symbol(&_hipMemAllocHost__funptr,"hipMemAllocHost")
    return (<hipError_t (*)(void **,unsigned long) nogil> _hipMemAllocHost__funptr)(ptr,size)


cdef void* _hipHostMalloc__funptr = NULL
# 
# @brief Allocate device accessible page locked host memory
# 
# @param[out] ptr Pointer to the allocated host pinned memory
# @param[in]  size Requested memory size
# @param[in]  flags Type of host memory allocation
# 
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# 
# @return #hipSuccess, #hipErrorOutOfMemory
# 
# @see hipSetDeviceFlags, hipHostFree
cdef hipError_t hipHostMalloc(void ** ptr,unsigned long size,unsigned int flags) nogil:
    global _hipHostMalloc__funptr
    __init_symbol(&_hipHostMalloc__funptr,"hipHostMalloc")
    return (<hipError_t (*)(void **,unsigned long,unsigned int) nogil> _hipHostMalloc__funptr)(ptr,size,flags)


cdef void* _hipMallocManaged__funptr = NULL
# 
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#   @addtogroup MemoryM Managed Memory
#   @{
#   @ingroup Memory
#   This section describes the managed memory management functions of HIP runtime API.
# 
# 
# 
#  @brief Allocates memory that will be automatically managed by HIP.
# 
#  @param [out] dev_ptr - pointer to allocated device memory
#  @param [in]  size    - requested allocation size in bytes
#  @param [in]  flags   - must be either hipMemAttachGlobal or hipMemAttachHost
#                         (defaults to hipMemAttachGlobal)
# 
#  @returns #hipSuccess, #hipErrorMemoryAllocation, #hipErrorNotSupported, #hipErrorInvalidValue
cdef hipError_t hipMallocManaged(void ** dev_ptr,unsigned long size,unsigned int flags) nogil:
    global _hipMallocManaged__funptr
    __init_symbol(&_hipMallocManaged__funptr,"hipMallocManaged")
    return (<hipError_t (*)(void **,unsigned long,unsigned int) nogil> _hipMallocManaged__funptr)(dev_ptr,size,flags)


cdef void* _hipMemPrefetchAsync__funptr = NULL
# 
# @brief Prefetches memory to the specified destination device using HIP.
# 
# @param [in] dev_ptr  pointer to be prefetched
# @param [in] count    size in bytes for prefetching
# @param [in] device   destination device to prefetch to
# @param [in] stream   stream to enqueue prefetch operation
# 
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemPrefetchAsync(const void * dev_ptr,unsigned long count,int device,hipStream_t stream) nogil:
    global _hipMemPrefetchAsync__funptr
    __init_symbol(&_hipMemPrefetchAsync__funptr,"hipMemPrefetchAsync")
    return (<hipError_t (*)(const void *,unsigned long,int,hipStream_t) nogil> _hipMemPrefetchAsync__funptr)(dev_ptr,count,device,stream)


cdef void* _hipMemAdvise__funptr = NULL
# 
# @brief Advise about the usage of a given memory range to HIP.
# 
# @param [in] dev_ptr  pointer to memory to set the advice for
# @param [in] count    size in bytes of the memory range
# @param [in] advice   advice to be applied for the specified memory range
# @param [in] device   device to apply the advice for
# 
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemAdvise(const void * dev_ptr,unsigned long count,hipMemoryAdvise advice,int device) nogil:
    global _hipMemAdvise__funptr
    __init_symbol(&_hipMemAdvise__funptr,"hipMemAdvise")
    return (<hipError_t (*)(const void *,unsigned long,hipMemoryAdvise,int) nogil> _hipMemAdvise__funptr)(dev_ptr,count,advice,device)


cdef void* _hipMemRangeGetAttribute__funptr = NULL
# 
# @brief Query an attribute of a given memory range in HIP.
# 
# @param [in,out] data   a pointer to a memory location where the result of each
#                        attribute query will be written to
# @param [in] data_size  the size of data
# @param [in] attribute  the attribute to query
# @param [in] dev_ptr    start of the range to query
# @param [in] count      size of the range to query
# 
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemRangeGetAttribute(void * data,unsigned long data_size,hipMemRangeAttribute attribute,const void * dev_ptr,unsigned long count) nogil:
    global _hipMemRangeGetAttribute__funptr
    __init_symbol(&_hipMemRangeGetAttribute__funptr,"hipMemRangeGetAttribute")
    return (<hipError_t (*)(void *,unsigned long,hipMemRangeAttribute,const void *,unsigned long) nogil> _hipMemRangeGetAttribute__funptr)(data,data_size,attribute,dev_ptr,count)


cdef void* _hipMemRangeGetAttributes__funptr = NULL
# 
# @brief Query attributes of a given memory range in HIP.
# 
# @param [in,out] data     a two-dimensional array containing pointers to memory locations
#                          where the result of each attribute query will be written to
# @param [in] data_sizes   an array, containing the sizes of each result
# @param [in] attributes   the attribute to query
# @param [in] num_attributes  an array of attributes to query (numAttributes and the number
#                          of attributes in this array should match)
# @param [in] dev_ptr      start of the range to query
# @param [in] count        size of the range to query
# 
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipMemRangeGetAttributes(void ** data,unsigned long * data_sizes,hipMemRangeAttribute * attributes,unsigned long num_attributes,const void * dev_ptr,unsigned long count) nogil:
    global _hipMemRangeGetAttributes__funptr
    __init_symbol(&_hipMemRangeGetAttributes__funptr,"hipMemRangeGetAttributes")
    return (<hipError_t (*)(void **,unsigned long *,hipMemRangeAttribute *,unsigned long,const void *,unsigned long) nogil> _hipMemRangeGetAttributes__funptr)(data,data_sizes,attributes,num_attributes,dev_ptr,count)


cdef void* _hipStreamAttachMemAsync__funptr = NULL
# 
# @brief Attach memory to a stream asynchronously in HIP.
# 
# @param [in] stream     - stream in which to enqueue the attach operation
# @param [in] dev_ptr    - pointer to memory (must be a pointer to managed memory or
#                          to a valid host-accessible region of system-allocated memory)
# @param [in] length     - length of memory (defaults to zero)
# @param [in] flags      - must be one of hipMemAttachGlobal, hipMemAttachHost or
#                          hipMemAttachSingle (defaults to hipMemAttachSingle)
# 
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipStreamAttachMemAsync(hipStream_t stream,void * dev_ptr,unsigned long length,unsigned int flags) nogil:
    global _hipStreamAttachMemAsync__funptr
    __init_symbol(&_hipStreamAttachMemAsync__funptr,"hipStreamAttachMemAsync")
    return (<hipError_t (*)(hipStream_t,void *,unsigned long,unsigned int) nogil> _hipStreamAttachMemAsync__funptr)(stream,dev_ptr,length,flags)


cdef void* _hipMallocAsync__funptr = NULL
# 
# @brief Allocates memory with stream ordered semantics
# 
# Inserts a memory allocation operation into @p stream.
# A pointer to the allocated memory is returned immediately in *dptr.
# The allocation must not be accessed until the the allocation operation completes.
# The allocation comes from the memory pool associated with the stream's device.
# 
# @note The default memory pool of a device contains device memory from that device.
# @note Basic stream ordering allows future work submitted into the same stream to use the allocation.
# Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
# operation completes before work submitted in a separate stream runs.
# @note During stream capture, this function results in the creation of an allocation node. In this case,
# the allocation is owned by the graph instead of the memory pool. The memory pool's properties
# are used to set the node's creation parameters.
# 
# @param [out] dev_ptr  Returned device pointer of memory allocation
# @param [in] size      Number of bytes to allocate
# @param [in] stream    The stream establishing the stream ordering contract and
#                       the memory pool to allocate from
# 
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
# 
# @see hipMallocFromPoolAsync, hipFreeAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMallocAsync(void ** dev_ptr,unsigned long size,hipStream_t stream) nogil:
    global _hipMallocAsync__funptr
    __init_symbol(&_hipMallocAsync__funptr,"hipMallocAsync")
    return (<hipError_t (*)(void **,unsigned long,hipStream_t) nogil> _hipMallocAsync__funptr)(dev_ptr,size,stream)


cdef void* _hipFreeAsync__funptr = NULL
# 
# @brief Frees memory with stream ordered semantics
# 
# Inserts a free operation into @p stream.
# The allocation must not be used after stream execution reaches the free.
# After this API returns, accessing the memory from any subsequent work launched on the GPU
# or querying its pointer attributes results in undefined behavior.
# 
# @note During stream capture, this function results in the creation of a free node and
# must therefore be passed the address of a graph allocation.
# 
# @param [in] dev_ptr Pointer to device memory to free
# @param [in] stream  The stream, where the destruciton will occur according to the execution order
# 
# @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported
# 
# @see hipMallocFromPoolAsync, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipFreeAsync(void * dev_ptr,hipStream_t stream) nogil:
    global _hipFreeAsync__funptr
    __init_symbol(&_hipFreeAsync__funptr,"hipFreeAsync")
    return (<hipError_t (*)(void *,hipStream_t) nogil> _hipFreeAsync__funptr)(dev_ptr,stream)


cdef void* _hipMemPoolTrimTo__funptr = NULL
# 
# @brief Releases freed memory back to the OS
# 
# Releases memory back to the OS until the pool contains fewer than @p min_bytes_to_keep
# reserved bytes, or there is no more memory that the allocator can safely release.
# The allocator cannot release OS allocations that back outstanding asynchronous allocations.
# The OS allocations may happen at different granularity from the user allocations.
# 
# @note: Allocations that have not been freed count as outstanding.
# @note: Allocations that have been asynchronously freed but whose completion has
# not been observed on the host (eg. by a synchronize) can count as outstanding.
# 
# @param[in] mem_pool          The memory pool to trim allocations
# @param[in] min_bytes_to_hold If the pool has less than min_bytes_to_hold reserved,
# then the TrimTo operation is a no-op.  Otherwise the memory pool will contain
# at least min_bytes_to_hold bytes reserved after the operation.
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
# hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool,unsigned long min_bytes_to_hold) nogil:
    global _hipMemPoolTrimTo__funptr
    __init_symbol(&_hipMemPoolTrimTo__funptr,"hipMemPoolTrimTo")
    return (<hipError_t (*)(hipMemPool_t,unsigned long) nogil> _hipMemPoolTrimTo__funptr)(mem_pool,min_bytes_to_hold)


cdef void* _hipMemPoolSetAttribute__funptr = NULL
# 
# @brief Sets attributes of a memory pool
# 
# Supported attributes are:
# - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
#                                  Amount of reserved memory in bytes to hold onto before trying
#                                  to release memory back to the OS. When more than the release
#                                  threshold bytes of memory are held by the memory pool, the
#                                  allocator will try to release memory back to the OS on the
#                                  next call to stream, event or context synchronize. (default 0)
# - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
#                                  Allow @p hipMallocAsync to use memory asynchronously freed
#                                  in another stream as long as a stream ordering dependency
#                                  of the allocating stream on the free action exists.
#                                  HIP events and null stream interactions can create the required
#                                  stream ordered dependencies. (default enabled)
# - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
#                                  Allow reuse of already completed frees when there is no dependency
#                                  between the free and allocation. (default enabled)
# - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
#                                  Allow @p hipMallocAsync to insert new stream dependencies
#                                  in order to establish the stream ordering required to reuse
#                                  a piece of memory released by @p hipFreeAsync (default enabled).
# 
# @param [in] mem_pool The memory pool to modify
# @param [in] attr     The attribute to modify
# @param [in] value    Pointer to the value to assign
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool,hipMemPoolAttr attr,void * value) nogil:
    global _hipMemPoolSetAttribute__funptr
    __init_symbol(&_hipMemPoolSetAttribute__funptr,"hipMemPoolSetAttribute")
    return (<hipError_t (*)(hipMemPool_t,hipMemPoolAttr,void *) nogil> _hipMemPoolSetAttribute__funptr)(mem_pool,attr,value)


cdef void* _hipMemPoolGetAttribute__funptr = NULL
# 
# @brief Gets attributes of a memory pool
# 
# Supported attributes are:
# - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
#                                  Amount of reserved memory in bytes to hold onto before trying
#                                  to release memory back to the OS. When more than the release
#                                  threshold bytes of memory are held by the memory pool, the
#                                  allocator will try to release memory back to the OS on the
#                                  next call to stream, event or context synchronize. (default 0)
# - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
#                                  Allow @p hipMallocAsync to use memory asynchronously freed
#                                  in another stream as long as a stream ordering dependency
#                                  of the allocating stream on the free action exists.
#                                  HIP events and null stream interactions can create the required
#                                  stream ordered dependencies. (default enabled)
# - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
#                                  Allow reuse of already completed frees when there is no dependency
#                                  between the free and allocation. (default enabled)
# - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
#                                  Allow @p hipMallocAsync to insert new stream dependencies
#                                  in order to establish the stream ordering required to reuse
#                                  a piece of memory released by @p hipFreeAsync (default enabled).
# 
# @param [in] mem_pool The memory pool to get attributes of
# @param [in] attr     The attribute to get
# @param [in] value    Retrieved value
# 
# @returns  #hipSuccess, #hipErrorInvalidValue
# 
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool,hipMemPoolAttr attr,void * value) nogil:
    global _hipMemPoolGetAttribute__funptr
    __init_symbol(&_hipMemPoolGetAttribute__funptr,"hipMemPoolGetAttribute")
    return (<hipError_t (*)(hipMemPool_t,hipMemPoolAttr,void *) nogil> _hipMemPoolGetAttribute__funptr)(mem_pool,attr,value)


cdef void* _hipMemPoolSetAccess__funptr = NULL
# 
# @brief Controls visibility of the specified pool between devices
# 
# @param [in] mem_pool   Memory pool for acccess change
# @param [in] desc_list  Array of access descriptors. Each descriptor instructs the access to enable for a single gpu
# @param [in] count  Number of descriptors in the map array.
# 
# @returns  #hipSuccess, #hipErrorInvalidValue
# 
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool,hipMemAccessDesc * desc_list,unsigned long count) nogil:
    global _hipMemPoolSetAccess__funptr
    __init_symbol(&_hipMemPoolSetAccess__funptr,"hipMemPoolSetAccess")
    return (<hipError_t (*)(hipMemPool_t,hipMemAccessDesc *,unsigned long) nogil> _hipMemPoolSetAccess__funptr)(mem_pool,desc_list,count)


cdef void* _hipMemPoolGetAccess__funptr = NULL
# 
# @brief Returns the accessibility of a pool from a device
# 
# Returns the accessibility of the pool's memory from the specified location.
# 
# @param [out] flags    Accessibility of the memory pool from the specified location/device
# @param [in] mem_pool   Memory pool being queried
# @param [in] location  Location/device for memory pool access
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolGetAccess(hipMemAccessFlags * flags,hipMemPool_t mem_pool,hipMemLocation * location) nogil:
    global _hipMemPoolGetAccess__funptr
    __init_symbol(&_hipMemPoolGetAccess__funptr,"hipMemPoolGetAccess")
    return (<hipError_t (*)(hipMemAccessFlags *,hipMemPool_t,hipMemLocation *) nogil> _hipMemPoolGetAccess__funptr)(flags,mem_pool,location)


cdef void* _hipMemPoolCreate__funptr = NULL
# 
# @brief Creates a memory pool
# 
# Creates a HIP memory pool and returns the handle in @p mem_pool. The @p pool_props determines
# the properties of the pool such as the backing device and IPC capabilities.
# 
# By default, the memory pool will be accessible from the device it is allocated on.
# 
# @param [out] mem_pool    Contains createed memory pool
# @param [in] pool_props   Memory pool properties
# 
# @note Specifying hipMemHandleTypeNone creates a memory pool that will not support IPC.
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# 
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolDestroy,
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool,hipMemPoolProps * pool_props) nogil:
    global _hipMemPoolCreate__funptr
    __init_symbol(&_hipMemPoolCreate__funptr,"hipMemPoolCreate")
    return (<hipError_t (*)(hipMemPool_t*,hipMemPoolProps *) nogil> _hipMemPoolCreate__funptr)(mem_pool,pool_props)


cdef void* _hipMemPoolDestroy__funptr = NULL
# 
# @brief Destroys the specified memory pool
# 
# If any pointers obtained from this pool haven't been freed or
# the pool has free operations that haven't completed
# when @p hipMemPoolDestroy is invoked, the function will return immediately and the
# resources associated with the pool will be released automatically
# once there are no more outstanding allocations.
# 
# Destroying the current mempool of a device sets the default mempool of
# that device as the current mempool for that device.
# 
# @param [in] mem_pool Memory pool for destruction
# 
# @note A device's default memory pool cannot be destroyed.
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) nogil:
    global _hipMemPoolDestroy__funptr
    __init_symbol(&_hipMemPoolDestroy__funptr,"hipMemPoolDestroy")
    return (<hipError_t (*)(hipMemPool_t) nogil> _hipMemPoolDestroy__funptr)(mem_pool)


cdef void* _hipMallocFromPoolAsync__funptr = NULL
# 
# @brief Allocates memory from a specified pool with stream ordered semantics.
# 
# Inserts an allocation operation into @p stream.
# A pointer to the allocated memory is returned immediately in @p dev_ptr.
# The allocation must not be accessed until the the allocation operation completes.
# The allocation comes from the specified memory pool.
# 
# @note The specified memory pool may be from a device different than that of the specified @p stream.
# 
# Basic stream ordering allows future work submitted into the same stream to use the allocation.
# Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
# operation completes before work submitted in a separate stream runs.
# 
# @note During stream capture, this function results in the creation of an allocation node. In this case,
# the allocation is owned by the graph instead of the memory pool. The memory pool's properties
# are used to set the node's creation parameters.
# 
# @param [out] dev_ptr Returned device pointer
# @param [in] size     Number of bytes to allocate
# @param [in] mem_pool The pool to allocate from
# @param [in] stream   The stream establishing the stream ordering semantic
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
# 
# @see hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
# hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess,
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMallocFromPoolAsync(void ** dev_ptr,unsigned long size,hipMemPool_t mem_pool,hipStream_t stream) nogil:
    global _hipMallocFromPoolAsync__funptr
    __init_symbol(&_hipMallocFromPoolAsync__funptr,"hipMallocFromPoolAsync")
    return (<hipError_t (*)(void **,unsigned long,hipMemPool_t,hipStream_t) nogil> _hipMallocFromPoolAsync__funptr)(dev_ptr,size,mem_pool,stream)


cdef void* _hipMemPoolExportToShareableHandle__funptr = NULL
# 
# @brief Exports a memory pool to the requested handle type.
# 
# Given an IPC capable mempool, create an OS handle to share the pool with another process.
# A recipient process can convert the shareable handle into a mempool with @p hipMemPoolImportFromShareableHandle.
# Individual pointers can then be shared with the @p hipMemPoolExportPointer and @p hipMemPoolImportPointer APIs.
# The implementation of what the shareable handle is and how it can be transferred is defined by the requested
# handle type.
# 
# @note: To create an IPC capable mempool, create a mempool with a @p hipMemAllocationHandleType other
# than @p hipMemHandleTypeNone.
# 
# @param [out] shared_handle Pointer to the location in which to store the requested handle
# @param [in] mem_pool       Pool to export
# @param [in] handle_type    The type of handle to create
# @param [in] flags          Must be 0
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
# 
# @see hipMemPoolImportFromShareableHandle
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolExportToShareableHandle(void * shared_handle,hipMemPool_t mem_pool,hipMemAllocationHandleType handle_type,unsigned int flags) nogil:
    global _hipMemPoolExportToShareableHandle__funptr
    __init_symbol(&_hipMemPoolExportToShareableHandle__funptr,"hipMemPoolExportToShareableHandle")
    return (<hipError_t (*)(void *,hipMemPool_t,hipMemAllocationHandleType,unsigned int) nogil> _hipMemPoolExportToShareableHandle__funptr)(shared_handle,mem_pool,handle_type,flags)


cdef void* _hipMemPoolImportFromShareableHandle__funptr = NULL
# 
# @brief Imports a memory pool from a shared handle.
# 
# Specific allocations can be imported from the imported pool with @p hipMemPoolImportPointer.
# 
# @note Imported memory pools do not support creating new allocations.
# As such imported memory pools may not be used in @p hipDeviceSetMemPool
# or @p hipMallocFromPoolAsync calls.
# 
# @param [out] mem_pool     Returned memory pool
# @param [in] shared_handle OS handle of the pool to open
# @param [in] handle_type   The type of handle being imported
# @param [in] flags         Must be 0
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
# 
# @see hipMemPoolExportToShareableHandle
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool,void * shared_handle,hipMemAllocationHandleType handle_type,unsigned int flags) nogil:
    global _hipMemPoolImportFromShareableHandle__funptr
    __init_symbol(&_hipMemPoolImportFromShareableHandle__funptr,"hipMemPoolImportFromShareableHandle")
    return (<hipError_t (*)(hipMemPool_t*,void *,hipMemAllocationHandleType,unsigned int) nogil> _hipMemPoolImportFromShareableHandle__funptr)(mem_pool,shared_handle,handle_type,flags)


cdef void* _hipMemPoolExportPointer__funptr = NULL
# 
# @brief Export data to share a memory pool allocation between processes.
# 
# Constructs @p export_data for sharing a specific allocation from an already shared memory pool.
# The recipient process can import the allocation with the @p hipMemPoolImportPointer api.
# The data is not a handle and may be shared through any IPC mechanism.
# 
# @param[out] export_data  Returned export data
# @param[in] dev_ptr       Pointer to memory being exported
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
# 
# @see hipMemPoolImportPointer
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData * export_data,void * dev_ptr) nogil:
    global _hipMemPoolExportPointer__funptr
    __init_symbol(&_hipMemPoolExportPointer__funptr,"hipMemPoolExportPointer")
    return (<hipError_t (*)(hipMemPoolPtrExportData *,void *) nogil> _hipMemPoolExportPointer__funptr)(export_data,dev_ptr)


cdef void* _hipMemPoolImportPointer__funptr = NULL
# 
# @brief Import a memory pool allocation from another process.
# 
# Returns in @p dev_ptr a pointer to the imported memory.
# The imported memory must not be accessed before the allocation operation completes
# in the exporting process. The imported memory must be freed from all importing processes before
# being freed in the exporting process. The pointer may be freed with @p hipFree
# or @p hipFreeAsync. If @p hipFreeAsync is used, the free must be completed
# on the importing process before the free operation on the exporting process.
# 
# @note The @p hipFreeAsync api may be used in the exporting process before
# the @p hipFreeAsync operation completes in its stream as long as the
# @p hipFreeAsync in the exporting process specifies a stream with
# a stream dependency on the importing process's @p hipFreeAsync.
# 
# @param [out] dev_ptr     Pointer to imported memory
# @param [in] mem_pool     Memory pool from which to import a pointer
# @param [in] export_data  Data specifying the memory to import
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized, #hipErrorOutOfMemory
# 
# @see hipMemPoolExportPointer
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemPoolImportPointer(void ** dev_ptr,hipMemPool_t mem_pool,hipMemPoolPtrExportData * export_data) nogil:
    global _hipMemPoolImportPointer__funptr
    __init_symbol(&_hipMemPoolImportPointer__funptr,"hipMemPoolImportPointer")
    return (<hipError_t (*)(void **,hipMemPool_t,hipMemPoolPtrExportData *) nogil> _hipMemPoolImportPointer__funptr)(dev_ptr,mem_pool,export_data)


cdef void* _hipHostAlloc__funptr = NULL
# 
# @brief Allocate device accessible page locked host memory [Deprecated]
# 
# @param[out] ptr Pointer to the allocated host pinned memory
# @param[in]  size Requested memory size
# @param[in]  flags Type of host memory allocation
# 
# If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# 
# @return #hipSuccess, #hipErrorOutOfMemory
# 
# @deprecated use hipHostMalloc() instead
cdef hipError_t hipHostAlloc(void ** ptr,unsigned long size,unsigned int flags) nogil:
    global _hipHostAlloc__funptr
    __init_symbol(&_hipHostAlloc__funptr,"hipHostAlloc")
    return (<hipError_t (*)(void **,unsigned long,unsigned int) nogil> _hipHostAlloc__funptr)(ptr,size,flags)


cdef void* _hipHostGetDevicePointer__funptr = NULL
# 
# @brief Get Device pointer from Host Pointer allocated through hipHostMalloc
# 
# @param[out] dstPtr Device Pointer mapped to passed host pointer
# @param[in]  hstPtr Host Pointer allocated through hipHostMalloc
# @param[in]  flags Flags to be passed for extension
# 
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
# 
# @see hipSetDeviceFlags, hipHostMalloc
cdef hipError_t hipHostGetDevicePointer(void ** devPtr,void * hstPtr,unsigned int flags) nogil:
    global _hipHostGetDevicePointer__funptr
    __init_symbol(&_hipHostGetDevicePointer__funptr,"hipHostGetDevicePointer")
    return (<hipError_t (*)(void **,void *,unsigned int) nogil> _hipHostGetDevicePointer__funptr)(devPtr,hstPtr,flags)


cdef void* _hipHostGetFlags__funptr = NULL
# 
# @brief Return flags associated with host pointer
# 
# @param[out] flagsPtr Memory location to store flags
# @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
# @return #hipSuccess, #hipErrorInvalidValue
# 
# @see hipHostMalloc
cdef hipError_t hipHostGetFlags(unsigned int * flagsPtr,void * hostPtr) nogil:
    global _hipHostGetFlags__funptr
    __init_symbol(&_hipHostGetFlags__funptr,"hipHostGetFlags")
    return (<hipError_t (*)(unsigned int *,void *) nogil> _hipHostGetFlags__funptr)(flagsPtr,hostPtr)


cdef void* _hipHostRegister__funptr = NULL
# 
#  @brief Register host memory so it can be accessed from the current device.
# 
#  @param[out] hostPtr Pointer to host memory to be registered.
#  @param[in] sizeBytes size of the host memory
#  @param[in] flags.  See below.
# 
#  Flags:
#  - #hipHostRegisterDefault   Memory is Mapped and Portable
#  - #hipHostRegisterPortable  Memory is considered registered by all contexts.  HIP only supports
# one context so this is always assumed true.
#  - #hipHostRegisterMapped    Map the allocation into the address space for the current device.
# The device pointer can be obtained with #hipHostGetDevicePointer.
# 
# 
#  After registering the memory, use #hipHostGetDevicePointer to obtain the mapped device pointer.
#  On many systems, the mapped device pointer will have a different value than the mapped host
# pointer.  Applications must use the device pointer in device code, and the host pointer in device
# code.
# 
#  On some systems, registered memory is pinned.  On some systems, registered memory may not be
# actually be pinned but uses OS or hardware facilities to all GPU access to the host memory.
# 
#  Developers are strongly encouraged to register memory blocks which are aligned to the host
# cache-line size. (typically 64-bytes but can be obtains from the CPUID instruction).
# 
#  If registering non-aligned pointers, the application must take care when register pointers from
# the same cache line on different devices.  HIP's coarse-grained synchronization model does not
# guarantee correct results if different devices write to different parts of the same cache block -
# typically one of the writes will "win" and overwrite data from the other registered memory
# region.
# 
#  @return #hipSuccess, #hipErrorOutOfMemory
# 
#  @see hipHostUnregister, hipHostGetFlags, hipHostGetDevicePointer
cdef hipError_t hipHostRegister(void * hostPtr,unsigned long sizeBytes,unsigned int flags) nogil:
    global _hipHostRegister__funptr
    __init_symbol(&_hipHostRegister__funptr,"hipHostRegister")
    return (<hipError_t (*)(void *,unsigned long,unsigned int) nogil> _hipHostRegister__funptr)(hostPtr,sizeBytes,flags)


cdef void* _hipHostUnregister__funptr = NULL
# 
# @brief Un-register host pointer
# 
# @param[in] hostPtr Host pointer previously registered with #hipHostRegister
# @return Error code
# 
# @see hipHostRegister
cdef hipError_t hipHostUnregister(void * hostPtr) nogil:
    global _hipHostUnregister__funptr
    __init_symbol(&_hipHostUnregister__funptr,"hipHostUnregister")
    return (<hipError_t (*)(void *) nogil> _hipHostUnregister__funptr)(hostPtr)


cdef void* _hipMallocPitch__funptr = NULL
# 
#  Allocates at least width (in bytes) * height bytes of linear memory
#  Padding may occur to ensure alighnment requirements are met for the given row
#  The change in width size due to padding will be returned in *pitch.
#  Currently the alignment is set to 128 bytes
# 
#  @param[out] ptr Pointer to the allocated device memory
#  @param[out] pitch Pitch for allocation (in bytes)
#  @param[in]  width Requested pitched allocation width (in bytes)
#  @param[in]  height Requested pitched allocation height
# 
#  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
# 
#  @return Error code
# 
#  @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
# hipMalloc3DArray, hipHostMalloc
cdef hipError_t hipMallocPitch(void ** ptr,unsigned long * pitch,unsigned long width,unsigned long height) nogil:
    global _hipMallocPitch__funptr
    __init_symbol(&_hipMallocPitch__funptr,"hipMallocPitch")
    return (<hipError_t (*)(void **,unsigned long *,unsigned long,unsigned long) nogil> _hipMallocPitch__funptr)(ptr,pitch,width,height)


cdef void* _hipMemAllocPitch__funptr = NULL
# 
#  Allocates at least width (in bytes) * height bytes of linear memory
#  Padding may occur to ensure alighnment requirements are met for the given row
#  The change in width size due to padding will be returned in *pitch.
#  Currently the alignment is set to 128 bytes
# 
#  @param[out] dptr Pointer to the allocated device memory
#  @param[out] pitch Pitch for allocation (in bytes)
#  @param[in]  width Requested pitched allocation width (in bytes)
#  @param[in]  height Requested pitched allocation height
# 
#  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
#  The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array.
#  Given the row and column of an array element of type T, the address is computed as:
#  T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
# 
#  @return Error code
# 
#  @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
# hipMalloc3DArray, hipHostMalloc
cdef hipError_t hipMemAllocPitch(void ** dptr,unsigned long * pitch,unsigned long widthInBytes,unsigned long height,unsigned int elementSizeBytes) nogil:
    global _hipMemAllocPitch__funptr
    __init_symbol(&_hipMemAllocPitch__funptr,"hipMemAllocPitch")
    return (<hipError_t (*)(void **,unsigned long *,unsigned long,unsigned long,unsigned int) nogil> _hipMemAllocPitch__funptr)(dptr,pitch,widthInBytes,height,elementSizeBytes)


cdef void* _hipFree__funptr = NULL
# 
#  @brief Free memory allocated by the hcc hip memory allocation API.
#  This API performs an implicit hipDeviceSynchronize() call.
#  If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
# 
#  @param[in] ptr Pointer to memory to be freed
#  @return #hipSuccess
#  @return #hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated
# with hipHostMalloc)
# 
#  @see hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
# hipMalloc3DArray, hipHostMalloc
cdef hipError_t hipFree(void * ptr) nogil:
    global _hipFree__funptr
    __init_symbol(&_hipFree__funptr,"hipFree")
    return (<hipError_t (*)(void *) nogil> _hipFree__funptr)(ptr)


cdef void* _hipFreeHost__funptr = NULL
# 
#    @brief Free memory allocated by the hcc hip host memory allocation API.  [Deprecated]
# 
#    @param[in] ptr Pointer to memory to be freed
#    @return #hipSuccess,
#            #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
# hipMalloc)
#    @deprecated use hipHostFree() instead
cdef hipError_t hipFreeHost(void * ptr) nogil:
    global _hipFreeHost__funptr
    __init_symbol(&_hipFreeHost__funptr,"hipFreeHost")
    return (<hipError_t (*)(void *) nogil> _hipFreeHost__funptr)(ptr)


cdef void* _hipHostFree__funptr = NULL
# 
#  @brief Free memory allocated by the hcc hip host memory allocation API
#  This API performs an implicit hipDeviceSynchronize() call.
#  If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
# 
#  @param[in] ptr Pointer to memory to be freed
#  @return #hipSuccess,
#          #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
# hipMalloc)
# 
#  @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D,
# hipMalloc3DArray, hipHostMalloc
cdef hipError_t hipHostFree(void * ptr) nogil:
    global _hipHostFree__funptr
    __init_symbol(&_hipHostFree__funptr,"hipHostFree")
    return (<hipError_t (*)(void *) nogil> _hipHostFree__funptr)(ptr)


cdef void* _hipMemcpy__funptr = NULL
# 
#  @brief Copy data from src to dst.
# 
#  It supports memory from host to device,
#  device to host, device to device and host to host
#  The src and dst must not overlap.
# 
#  For hipMemcpy, the copy is always performed by the current device (set by hipSetDevice).
#  For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the
#  device where the src data is physically located. For optimal peer-to-peer copies, the copy device
#  must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy
#  agent as the current device and src/dest as the peerDevice argument.  if this is not done, the
#  hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
#  Calling hipMemcpy with dst and src pointers that do not match the hipMemcpyKind results in
#  undefined behavior.
# 
#  @param[out]  dst Data being copy to
#  @param[in]  src Data being copy from
#  @param[in]  sizeBytes Data size in bytes
#  @param[in]  copyType Memory copy type
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknowni
# 
#  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpy(void * dst,const void * src,unsigned long sizeBytes,hipMemcpyKind kind) nogil:
    global _hipMemcpy__funptr
    __init_symbol(&_hipMemcpy__funptr,"hipMemcpy")
    return (<hipError_t (*)(void *,const void *,unsigned long,hipMemcpyKind) nogil> _hipMemcpy__funptr)(dst,src,sizeBytes,kind)


cdef void* _hipMemcpyWithStream__funptr = NULL
cdef hipError_t hipMemcpyWithStream(void * dst,const void * src,unsigned long sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpyWithStream__funptr
    __init_symbol(&_hipMemcpyWithStream__funptr,"hipMemcpyWithStream")
    return (<hipError_t (*)(void *,const void *,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpyWithStream__funptr)(dst,src,sizeBytes,kind,stream)


cdef void* _hipMemcpyHtoD__funptr = NULL
# 
#  @brief Copy data from Host to Device
# 
#  @param[out]  dst Data being copy to
#  @param[in]   src Data being copy from
#  @param[in]   sizeBytes Data size in bytes
# 
#  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# 
#  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyHtoD(void * dst,void * src,unsigned long sizeBytes) nogil:
    global _hipMemcpyHtoD__funptr
    __init_symbol(&_hipMemcpyHtoD__funptr,"hipMemcpyHtoD")
    return (<hipError_t (*)(void *,void *,unsigned long) nogil> _hipMemcpyHtoD__funptr)(dst,src,sizeBytes)


cdef void* _hipMemcpyDtoH__funptr = NULL
# 
#  @brief Copy data from Device to Host
# 
#  @param[out]  dst Data being copy to
#  @param[in]   src Data being copy from
#  @param[in]   sizeBytes Data size in bytes
# 
#  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# 
#  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyDtoH(void * dst,void * src,unsigned long sizeBytes) nogil:
    global _hipMemcpyDtoH__funptr
    __init_symbol(&_hipMemcpyDtoH__funptr,"hipMemcpyDtoH")
    return (<hipError_t (*)(void *,void *,unsigned long) nogil> _hipMemcpyDtoH__funptr)(dst,src,sizeBytes)


cdef void* _hipMemcpyDtoD__funptr = NULL
# 
#  @brief Copy data from Device to Device
# 
#  @param[out]  dst Data being copy to
#  @param[in]   src Data being copy from
#  @param[in]   sizeBytes Data size in bytes
# 
#  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# 
#  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyDtoD(void * dst,void * src,unsigned long sizeBytes) nogil:
    global _hipMemcpyDtoD__funptr
    __init_symbol(&_hipMemcpyDtoD__funptr,"hipMemcpyDtoD")
    return (<hipError_t (*)(void *,void *,unsigned long) nogil> _hipMemcpyDtoD__funptr)(dst,src,sizeBytes)


cdef void* _hipMemcpyHtoDAsync__funptr = NULL
# 
#  @brief Copy data from Host to Device asynchronously
# 
#  @param[out]  dst Data being copy to
#  @param[in]   src Data being copy from
#  @param[in]   sizeBytes Data size in bytes
# 
#  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# 
#  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyHtoDAsync(void * dst,void * src,unsigned long sizeBytes,hipStream_t stream) nogil:
    global _hipMemcpyHtoDAsync__funptr
    __init_symbol(&_hipMemcpyHtoDAsync__funptr,"hipMemcpyHtoDAsync")
    return (<hipError_t (*)(void *,void *,unsigned long,hipStream_t) nogil> _hipMemcpyHtoDAsync__funptr)(dst,src,sizeBytes,stream)


cdef void* _hipMemcpyDtoHAsync__funptr = NULL
# 
#  @brief Copy data from Device to Host asynchronously
# 
#  @param[out]  dst Data being copy to
#  @param[in]   src Data being copy from
#  @param[in]   sizeBytes Data size in bytes
# 
#  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# 
#  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyDtoHAsync(void * dst,void * src,unsigned long sizeBytes,hipStream_t stream) nogil:
    global _hipMemcpyDtoHAsync__funptr
    __init_symbol(&_hipMemcpyDtoHAsync__funptr,"hipMemcpyDtoHAsync")
    return (<hipError_t (*)(void *,void *,unsigned long,hipStream_t) nogil> _hipMemcpyDtoHAsync__funptr)(dst,src,sizeBytes,stream)


cdef void* _hipMemcpyDtoDAsync__funptr = NULL
# 
#  @brief Copy data from Device to Device asynchronously
# 
#  @param[out]  dst Data being copy to
#  @param[in]   src Data being copy from
#  @param[in]   sizeBytes Data size in bytes
# 
#  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
# #hipErrorInvalidValue
# 
#  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
# hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
# hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
# hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
# hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
# hipMemHostAlloc, hipMemHostGetDevicePointer
cdef hipError_t hipMemcpyDtoDAsync(void * dst,void * src,unsigned long sizeBytes,hipStream_t stream) nogil:
    global _hipMemcpyDtoDAsync__funptr
    __init_symbol(&_hipMemcpyDtoDAsync__funptr,"hipMemcpyDtoDAsync")
    return (<hipError_t (*)(void *,void *,unsigned long,hipStream_t) nogil> _hipMemcpyDtoDAsync__funptr)(dst,src,sizeBytes,stream)


cdef void* _hipModuleGetGlobal__funptr = NULL
# 
# @brief Returns a global pointer from a module.
# Returns in *dptr and *bytes the pointer and size of the global of name name located in module hmod.
# If no variable of that name exists, it returns hipErrorNotFound. Both parameters dptr and bytes are optional.
# If one of them is NULL, it is ignored and hipSuccess is returned.
# 
# @param[out]  dptr  Returns global device pointer
# @param[out]  bytes Returns global size in bytes
# @param[in]   hmod  Module to retrieve global from
# @param[in]   name  Name of global to retrieve
# 
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotFound, #hipErrorInvalidContext
#
cdef hipError_t hipModuleGetGlobal(void ** dptr,unsigned long * bytes,hipModule_t hmod,const char * name) nogil:
    global _hipModuleGetGlobal__funptr
    __init_symbol(&_hipModuleGetGlobal__funptr,"hipModuleGetGlobal")
    return (<hipError_t (*)(void **,unsigned long *,hipModule_t,const char *) nogil> _hipModuleGetGlobal__funptr)(dptr,bytes,hmod,name)


cdef void* _hipGetSymbolAddress__funptr = NULL
# 
# @brief Gets device pointer associated with symbol on the device.
# 
# @param[out]  devPtr  pointer to the device associated the symbole
# @param[in]   symbol  pointer to the symbole of the device
# 
# @return #hipSuccess, #hipErrorInvalidValue
#
cdef hipError_t hipGetSymbolAddress(void ** devPtr,const void * symbol) nogil:
    global _hipGetSymbolAddress__funptr
    __init_symbol(&_hipGetSymbolAddress__funptr,"hipGetSymbolAddress")
    return (<hipError_t (*)(void **,const void *) nogil> _hipGetSymbolAddress__funptr)(devPtr,symbol)


cdef void* _hipGetSymbolSize__funptr = NULL
# 
# @brief Gets the size of the given symbol on the device.
# 
# @param[in]   symbol  pointer to the device symbole
# @param[out]  size  pointer to the size
# 
# @return #hipSuccess, #hipErrorInvalidValue
#
cdef hipError_t hipGetSymbolSize(unsigned long * size,const void * symbol) nogil:
    global _hipGetSymbolSize__funptr
    __init_symbol(&_hipGetSymbolSize__funptr,"hipGetSymbolSize")
    return (<hipError_t (*)(unsigned long *,const void *) nogil> _hipGetSymbolSize__funptr)(size,symbol)


cdef void* _hipMemcpyToSymbol__funptr = NULL
# 
#  @brief Copies data to the given symbol on the device.
# Symbol HIP APIs allow a kernel to define a device-side data symbol which can be accessed on
# the host side. The symbol can be in __constant or device space.
# Note that the symbol name needs to be encased in the HIP_SYMBOL macro.
# This also applies to hipMemcpyFromSymbol, hipGetSymbolAddress, and hipGetSymbolSize.
# For detail usage, see the example at
# https://github.com/ROCm-Developer-Tools/HIP/blob/rocm-5.0.x/docs/markdown/hip_porting_guide.md
# 
#  @param[out]  symbol  pointer to the device symbole
#  @param[in]   src  pointer to the source address
#  @param[in]   sizeBytes  size in bytes to copy
#  @param[in]   offset  offset in bytes from start of symbole
#  @param[in]   kind  type of memory transfer
# 
#  @return #hipSuccess, #hipErrorInvalidValue
#
cdef hipError_t hipMemcpyToSymbol(const void * symbol,const void * src,unsigned long sizeBytes,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipMemcpyToSymbol__funptr
    __init_symbol(&_hipMemcpyToSymbol__funptr,"hipMemcpyToSymbol")
    return (<hipError_t (*)(const void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpyToSymbol__funptr)(symbol,src,sizeBytes,offset,kind)


cdef void* _hipMemcpyToSymbolAsync__funptr = NULL
# 
# @brief Copies data to the given symbol on the device asynchronously.
# 
# @param[out]  symbol  pointer to the device symbole
# @param[in]   src  pointer to the source address
# @param[in]   sizeBytes  size in bytes to copy
# @param[in]   offset  offset in bytes from start of symbole
# @param[in]   kind  type of memory transfer
# @param[in]   stream  stream identifier
# 
# @return #hipSuccess, #hipErrorInvalidValue
#
cdef hipError_t hipMemcpyToSymbolAsync(const void * symbol,const void * src,unsigned long sizeBytes,unsigned long offset,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpyToSymbolAsync__funptr
    __init_symbol(&_hipMemcpyToSymbolAsync__funptr,"hipMemcpyToSymbolAsync")
    return (<hipError_t (*)(const void *,const void *,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpyToSymbolAsync__funptr)(symbol,src,sizeBytes,offset,kind,stream)


cdef void* _hipMemcpyFromSymbol__funptr = NULL
# 
# @brief Copies data from the given symbol on the device.
# 
# @param[out]  dptr  Returns pointer to destinition memory address
# @param[in]   symbol  pointer to the symbole address on the device
# @param[in]   sizeBytes  size in bytes to copy
# @param[in]   offset  offset in bytes from the start of symbole
# @param[in]   kind  type of memory transfer
# 
# @return #hipSuccess, #hipErrorInvalidValue
#
cdef hipError_t hipMemcpyFromSymbol(void * dst,const void * symbol,unsigned long sizeBytes,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipMemcpyFromSymbol__funptr
    __init_symbol(&_hipMemcpyFromSymbol__funptr,"hipMemcpyFromSymbol")
    return (<hipError_t (*)(void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpyFromSymbol__funptr)(dst,symbol,sizeBytes,offset,kind)


cdef void* _hipMemcpyFromSymbolAsync__funptr = NULL
# 
# @brief Copies data from the given symbol on the device asynchronously.
# 
# @param[out]  dptr  Returns pointer to destinition memory address
# @param[in]   symbol  pointer to the symbole address on the device
# @param[in]   sizeBytes  size in bytes to copy
# @param[in]   offset  offset in bytes from the start of symbole
# @param[in]   kind  type of memory transfer
# @param[in]   stream  stream identifier
# 
# @return #hipSuccess, #hipErrorInvalidValue
#
cdef hipError_t hipMemcpyFromSymbolAsync(void * dst,const void * symbol,unsigned long sizeBytes,unsigned long offset,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpyFromSymbolAsync__funptr
    __init_symbol(&_hipMemcpyFromSymbolAsync__funptr,"hipMemcpyFromSymbolAsync")
    return (<hipError_t (*)(void *,const void *,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpyFromSymbolAsync__funptr)(dst,symbol,sizeBytes,offset,kind,stream)


cdef void* _hipMemcpyAsync__funptr = NULL
# 
#  @brief Copy data from src to dst asynchronously.
# 
#  @warning If host or dest are not pinned, the memory copy will be performed synchronously.  For
# best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.
# 
#  @warning on HCC hipMemcpyAsync does not support overlapped H2D and D2H copies.
#  For hipMemcpy, the copy is always performed by the device associated with the specified stream.
# 
#  For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is a
# attached to the device where the src data is physically located. For optimal peer-to-peer copies,
# the copy device must be able to access the src and dst pointers (by calling
# hipDeviceEnablePeerAccess with copy agent as the current device and src/dest as the peerDevice
# argument.  if this is not done, the hipMemcpy will still work, but will perform the copy using a
# staging buffer on the host.
# 
#  @param[out] dst Data being copy to
#  @param[in]  src Data being copy from
#  @param[in]  sizeBytes Data size in bytes
#  @param[in]  accelerator_view Accelerator view which the copy is being enqueued
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
# 
#  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
# hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyToSymbol,
# hipMemcpyFromSymbol, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync,
# hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync,
# hipMemcpyFromSymbolAsync
cdef hipError_t hipMemcpyAsync(void * dst,const void * src,unsigned long sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpyAsync__funptr
    __init_symbol(&_hipMemcpyAsync__funptr,"hipMemcpyAsync")
    return (<hipError_t (*)(void *,const void *,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpyAsync__funptr)(dst,src,sizeBytes,kind,stream)


cdef void* _hipMemset__funptr = NULL
# 
#  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# byte value value.
# 
#  @param[out] dst Data being filled
#  @param[in]  constant value to be set
#  @param[in]  sizeBytes Data size in bytes
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemset(void * dst,int value,unsigned long sizeBytes) nogil:
    global _hipMemset__funptr
    __init_symbol(&_hipMemset__funptr,"hipMemset")
    return (<hipError_t (*)(void *,int,unsigned long) nogil> _hipMemset__funptr)(dst,value,sizeBytes)


cdef void* _hipMemsetD8__funptr = NULL
# 
#  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# byte value value.
# 
#  @param[out] dst Data ptr to be filled
#  @param[in]  constant value to be set
#  @param[in]  number of values to be set
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD8(void * dest,unsigned char value,unsigned long count) nogil:
    global _hipMemsetD8__funptr
    __init_symbol(&_hipMemsetD8__funptr,"hipMemsetD8")
    return (<hipError_t (*)(void *,unsigned char,unsigned long) nogil> _hipMemsetD8__funptr)(dest,value,count)


cdef void* _hipMemsetD8Async__funptr = NULL
# 
#  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# byte value value.
# 
# hipMemsetD8Async() is asynchronous with respect to the host, so the call may return before the
# memset is complete. The operation can optionally be associated to a stream by passing a non-zero
# stream argument. If stream is non-zero, the operation may overlap with operations in other
# streams.
# 
#  @param[out] dst Data ptr to be filled
#  @param[in]  constant value to be set
#  @param[in]  number of values to be set
#  @param[in]  stream - Stream identifier
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD8Async(void * dest,unsigned char value,unsigned long count,hipStream_t stream) nogil:
    global _hipMemsetD8Async__funptr
    __init_symbol(&_hipMemsetD8Async__funptr,"hipMemsetD8Async")
    return (<hipError_t (*)(void *,unsigned char,unsigned long,hipStream_t) nogil> _hipMemsetD8Async__funptr)(dest,value,count,stream)


cdef void* _hipMemsetD16__funptr = NULL
# 
#  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# short value value.
# 
#  @param[out] dst Data ptr to be filled
#  @param[in]  constant value to be set
#  @param[in]  number of values to be set
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD16(void * dest,unsigned short value,unsigned long count) nogil:
    global _hipMemsetD16__funptr
    __init_symbol(&_hipMemsetD16__funptr,"hipMemsetD16")
    return (<hipError_t (*)(void *,unsigned short,unsigned long) nogil> _hipMemsetD16__funptr)(dest,value,count)


cdef void* _hipMemsetD16Async__funptr = NULL
# 
#  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
# short value value.
# 
# hipMemsetD16Async() is asynchronous with respect to the host, so the call may return before the
# memset is complete. The operation can optionally be associated to a stream by passing a non-zero
# stream argument. If stream is non-zero, the operation may overlap with operations in other
# streams.
# 
#  @param[out] dst Data ptr to be filled
#  @param[in]  constant value to be set
#  @param[in]  number of values to be set
#  @param[in]  stream - Stream identifier
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD16Async(void * dest,unsigned short value,unsigned long count,hipStream_t stream) nogil:
    global _hipMemsetD16Async__funptr
    __init_symbol(&_hipMemsetD16Async__funptr,"hipMemsetD16Async")
    return (<hipError_t (*)(void *,unsigned short,unsigned long,hipStream_t) nogil> _hipMemsetD16Async__funptr)(dest,value,count,stream)


cdef void* _hipMemsetD32__funptr = NULL
# 
#  @brief Fills the memory area pointed to by dest with the constant integer
# value for specified number of times.
# 
#  @param[out] dst Data being filled
#  @param[in]  constant value to be set
#  @param[in]  number of values to be set
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
cdef hipError_t hipMemsetD32(void * dest,int value,unsigned long count) nogil:
    global _hipMemsetD32__funptr
    __init_symbol(&_hipMemsetD32__funptr,"hipMemsetD32")
    return (<hipError_t (*)(void *,int,unsigned long) nogil> _hipMemsetD32__funptr)(dest,value,count)


cdef void* _hipMemsetAsync__funptr = NULL
# 
#  @brief Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant
# byte value value.
# 
#  hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the
# memset is complete. The operation can optionally be associated to a stream by passing a non-zero
# stream argument. If stream is non-zero, the operation may overlap with operations in other
# streams.
# 
#  @param[out] dst Pointer to device memory
#  @param[in]  value - Value to set for each byte of specified memory
#  @param[in]  sizeBytes - Size in bytes to set
#  @param[in]  stream - Stream identifier
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemsetAsync(void * dst,int value,unsigned long sizeBytes,hipStream_t stream) nogil:
    global _hipMemsetAsync__funptr
    __init_symbol(&_hipMemsetAsync__funptr,"hipMemsetAsync")
    return (<hipError_t (*)(void *,int,unsigned long,hipStream_t) nogil> _hipMemsetAsync__funptr)(dst,value,sizeBytes,stream)


cdef void* _hipMemsetD32Async__funptr = NULL
# 
#  @brief Fills the memory area pointed to by dev with the constant integer
# value for specified number of times.
# 
#  hipMemsetD32Async() is asynchronous with respect to the host, so the call may return before the
# memset is complete. The operation can optionally be associated to a stream by passing a non-zero
# stream argument. If stream is non-zero, the operation may overlap with operations in other
# streams.
# 
#  @param[out] dst Pointer to device memory
#  @param[in]  value - Value to set for each byte of specified memory
#  @param[in]  count - number of values to be set
#  @param[in]  stream - Stream identifier
#  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemsetD32Async(void * dst,int value,unsigned long count,hipStream_t stream) nogil:
    global _hipMemsetD32Async__funptr
    __init_symbol(&_hipMemsetD32Async__funptr,"hipMemsetD32Async")
    return (<hipError_t (*)(void *,int,unsigned long,hipStream_t) nogil> _hipMemsetD32Async__funptr)(dst,value,count,stream)


cdef void* _hipMemset2D__funptr = NULL
# 
# @brief Fills the memory area pointed to by dst with the constant value.
# 
# @param[out] dst Pointer to device memory
# @param[in]  pitch - data size in bytes
# @param[in]  value - constant value to be set
# @param[in]  width
# @param[in]  height
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemset2D(void * dst,unsigned long pitch,int value,unsigned long width,unsigned long height) nogil:
    global _hipMemset2D__funptr
    __init_symbol(&_hipMemset2D__funptr,"hipMemset2D")
    return (<hipError_t (*)(void *,unsigned long,int,unsigned long,unsigned long) nogil> _hipMemset2D__funptr)(dst,pitch,value,width,height)


cdef void* _hipMemset2DAsync__funptr = NULL
# 
# @brief Fills asynchronously the memory area pointed to by dst with the constant value.
# 
# @param[in]  dst Pointer to device memory
# @param[in]  pitch - data size in bytes
# @param[in]  value - constant value to be set
# @param[in]  width
# @param[in]  height
# @param[in]  stream
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemset2DAsync(void * dst,unsigned long pitch,int value,unsigned long width,unsigned long height,hipStream_t stream) nogil:
    global _hipMemset2DAsync__funptr
    __init_symbol(&_hipMemset2DAsync__funptr,"hipMemset2DAsync")
    return (<hipError_t (*)(void *,unsigned long,int,unsigned long,unsigned long,hipStream_t) nogil> _hipMemset2DAsync__funptr)(dst,pitch,value,width,height,stream)


cdef void* _hipMemset3D__funptr = NULL
# 
# @brief Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
# 
# @param[in] pitchedDevPtr
# @param[in]  value - constant value to be set
# @param[in]  extent
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent) nogil:
    global _hipMemset3D__funptr
    __init_symbol(&_hipMemset3D__funptr,"hipMemset3D")
    return (<hipError_t (*)(hipPitchedPtr,int,hipExtent) nogil> _hipMemset3D__funptr)(pitchedDevPtr,value,extent)


cdef void* _hipMemset3DAsync__funptr = NULL
# 
# @brief Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
# 
# @param[in] pitchedDevPtr
# @param[in]  value - constant value to be set
# @param[in]  extent
# @param[in]  stream
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
cdef hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent,hipStream_t stream) nogil:
    global _hipMemset3DAsync__funptr
    __init_symbol(&_hipMemset3DAsync__funptr,"hipMemset3DAsync")
    return (<hipError_t (*)(hipPitchedPtr,int,hipExtent,hipStream_t) nogil> _hipMemset3DAsync__funptr)(pitchedDevPtr,value,extent,stream)


cdef void* _hipMemGetInfo__funptr = NULL
# 
#  @brief Query memory info.
#  Return snapshot of free memory, and total allocatable memory on the device.
# 
#  Returns in *free a snapshot of the current free memory.
#  @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
#  @warning On HCC, the free memory only accounts for memory allocated by this process and may be
# optimistic.
cdef hipError_t hipMemGetInfo(unsigned long * free,unsigned long * total) nogil:
    global _hipMemGetInfo__funptr
    __init_symbol(&_hipMemGetInfo__funptr,"hipMemGetInfo")
    return (<hipError_t (*)(unsigned long *,unsigned long *) nogil> _hipMemGetInfo__funptr)(free,total)


cdef void* _hipMemPtrGetInfo__funptr = NULL
cdef hipError_t hipMemPtrGetInfo(void * ptr,unsigned long * size) nogil:
    global _hipMemPtrGetInfo__funptr
    __init_symbol(&_hipMemPtrGetInfo__funptr,"hipMemPtrGetInfo")
    return (<hipError_t (*)(void *,unsigned long *) nogil> _hipMemPtrGetInfo__funptr)(ptr,size)


cdef void* _hipMallocArray__funptr = NULL
# 
# @brief Allocate an array on the device.
# 
# @param[out]  array  Pointer to allocated array in device memory
# @param[in]   desc   Requested channel format
# @param[in]   width  Requested array allocation width
# @param[in]   height Requested array allocation height
# @param[in]   flags  Requested properties of allocated array
# @return      #hipSuccess, #hipErrorOutOfMemory
# 
# @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
cdef hipError_t hipMallocArray(hipArray ** array,hipChannelFormatDesc * desc,unsigned long width,unsigned long height,unsigned int flags) nogil:
    global _hipMallocArray__funptr
    __init_symbol(&_hipMallocArray__funptr,"hipMallocArray")
    return (<hipError_t (*)(hipArray **,hipChannelFormatDesc *,unsigned long,unsigned long,unsigned int) nogil> _hipMallocArray__funptr)(array,desc,width,height,flags)


cdef void* _hipArrayCreate__funptr = NULL
cdef hipError_t hipArrayCreate(hipArray ** pHandle,HIP_ARRAY_DESCRIPTOR * pAllocateArray) nogil:
    global _hipArrayCreate__funptr
    __init_symbol(&_hipArrayCreate__funptr,"hipArrayCreate")
    return (<hipError_t (*)(hipArray **,HIP_ARRAY_DESCRIPTOR *) nogil> _hipArrayCreate__funptr)(pHandle,pAllocateArray)


cdef void* _hipArrayDestroy__funptr = NULL
cdef hipError_t hipArrayDestroy(hipArray * array) nogil:
    global _hipArrayDestroy__funptr
    __init_symbol(&_hipArrayDestroy__funptr,"hipArrayDestroy")
    return (<hipError_t (*)(hipArray *) nogil> _hipArrayDestroy__funptr)(array)


cdef void* _hipArray3DCreate__funptr = NULL
cdef hipError_t hipArray3DCreate(hipArray ** array,HIP_ARRAY3D_DESCRIPTOR * pAllocateArray) nogil:
    global _hipArray3DCreate__funptr
    __init_symbol(&_hipArray3DCreate__funptr,"hipArray3DCreate")
    return (<hipError_t (*)(hipArray **,HIP_ARRAY3D_DESCRIPTOR *) nogil> _hipArray3DCreate__funptr)(array,pAllocateArray)


cdef void* _hipMalloc3D__funptr = NULL
cdef hipError_t hipMalloc3D(hipPitchedPtr * pitchedDevPtr,hipExtent extent) nogil:
    global _hipMalloc3D__funptr
    __init_symbol(&_hipMalloc3D__funptr,"hipMalloc3D")
    return (<hipError_t (*)(hipPitchedPtr *,hipExtent) nogil> _hipMalloc3D__funptr)(pitchedDevPtr,extent)


cdef void* _hipFreeArray__funptr = NULL
# 
# @brief Frees an array on the device.
# 
# @param[in]  array  Pointer to array to free
# @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
# 
# @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
cdef hipError_t hipFreeArray(hipArray * array) nogil:
    global _hipFreeArray__funptr
    __init_symbol(&_hipFreeArray__funptr,"hipFreeArray")
    return (<hipError_t (*)(hipArray *) nogil> _hipFreeArray__funptr)(array)


cdef void* _hipFreeMipmappedArray__funptr = NULL
# 
# @brief Frees a mipmapped array on the device
# 
# @param[in] mipmappedArray - Pointer to mipmapped array to free
# 
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) nogil:
    global _hipFreeMipmappedArray__funptr
    __init_symbol(&_hipFreeMipmappedArray__funptr,"hipFreeMipmappedArray")
    return (<hipError_t (*)(hipMipmappedArray_t) nogil> _hipFreeMipmappedArray__funptr)(mipmappedArray)


cdef void* _hipMalloc3DArray__funptr = NULL
# 
# @brief Allocate an array on the device.
# 
# @param[out]  array  Pointer to allocated array in device memory
# @param[in]   desc   Requested channel format
# @param[in]   extent Requested array allocation width, height and depth
# @param[in]   flags  Requested properties of allocated array
# @return      #hipSuccess, #hipErrorOutOfMemory
# 
# @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
cdef hipError_t hipMalloc3DArray(hipArray ** array,hipChannelFormatDesc * desc,hipExtent extent,unsigned int flags) nogil:
    global _hipMalloc3DArray__funptr
    __init_symbol(&_hipMalloc3DArray__funptr,"hipMalloc3DArray")
    return (<hipError_t (*)(hipArray **,hipChannelFormatDesc *,hipExtent,unsigned int) nogil> _hipMalloc3DArray__funptr)(array,desc,extent,flags)


cdef void* _hipMallocMipmappedArray__funptr = NULL
# 
# @brief Allocate a mipmapped array on the device
# 
# @param[out] mipmappedArray  - Pointer to allocated mipmapped array in device memory
# @param[in]  desc            - Requested channel format
# @param[in]  extent          - Requested allocation size (width field in elements)
# @param[in]  numLevels       - Number of mipmap levels to allocate
# @param[in]  flags           - Flags for extensions
# 
# @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
cdef hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray,hipChannelFormatDesc * desc,hipExtent extent,unsigned int numLevels,unsigned int flags) nogil:
    global _hipMallocMipmappedArray__funptr
    __init_symbol(&_hipMallocMipmappedArray__funptr,"hipMallocMipmappedArray")
    return (<hipError_t (*)(hipMipmappedArray_t*,hipChannelFormatDesc *,hipExtent,unsigned int,unsigned int) nogil> _hipMallocMipmappedArray__funptr)(mipmappedArray,desc,extent,numLevels,flags)


cdef void* _hipGetMipmappedArrayLevel__funptr = NULL
# 
# @brief Gets a mipmap level of a HIP mipmapped array
# 
# @param[out] levelArray     - Returned mipmap level HIP array
# @param[in]  mipmappedArray - HIP mipmapped array
# @param[in]  level          - Mipmap level
# 
# @return #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray,hipMipmappedArray_const_t mipmappedArray,unsigned int level) nogil:
    global _hipGetMipmappedArrayLevel__funptr
    __init_symbol(&_hipGetMipmappedArrayLevel__funptr,"hipGetMipmappedArrayLevel")
    return (<hipError_t (*)(hipArray_t*,hipMipmappedArray_const_t,unsigned int) nogil> _hipGetMipmappedArrayLevel__funptr)(levelArray,mipmappedArray,level)


cdef void* _hipMemcpy2D__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   dst    Destination memory address
#  @param[in]   dpitch Pitch of destination memory
#  @param[in]   src    Source memory address
#  @param[in]   spitch Pitch of source memory
#  @param[in]   width  Width of matrix transfer (columns in bytes)
#  @param[in]   height Height of matrix transfer (rows)
#  @param[in]   kind   Type of transfer
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2D(void * dst,unsigned long dpitch,const void * src,unsigned long spitch,unsigned long width,unsigned long height,hipMemcpyKind kind) nogil:
    global _hipMemcpy2D__funptr
    __init_symbol(&_hipMemcpy2D__funptr,"hipMemcpy2D")
    return (<hipError_t (*)(void *,unsigned long,const void *,unsigned long,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpy2D__funptr)(dst,dpitch,src,spitch,width,height,kind)


cdef void* _hipMemcpyParam2D__funptr = NULL
# 
#  @brief Copies memory for 2D arrays.
#  @param[in]   pCopy Parameters for the memory copy
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
#  #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
# hipMemcpyToSymbol, hipMemcpyAsync
cdef hipError_t hipMemcpyParam2D(hip_Memcpy2D * pCopy) nogil:
    global _hipMemcpyParam2D__funptr
    __init_symbol(&_hipMemcpyParam2D__funptr,"hipMemcpyParam2D")
    return (<hipError_t (*)(hip_Memcpy2D *) nogil> _hipMemcpyParam2D__funptr)(pCopy)


cdef void* _hipMemcpyParam2DAsync__funptr = NULL
# 
#  @brief Copies memory for 2D arrays.
#  @param[in]   pCopy Parameters for the memory copy
#  @param[in]   stream Stream to use
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
# hipMemcpyToSymbol, hipMemcpyAsync
cdef hipError_t hipMemcpyParam2DAsync(hip_Memcpy2D * pCopy,hipStream_t stream) nogil:
    global _hipMemcpyParam2DAsync__funptr
    __init_symbol(&_hipMemcpyParam2DAsync__funptr,"hipMemcpyParam2DAsync")
    return (<hipError_t (*)(hip_Memcpy2D *,hipStream_t) nogil> _hipMemcpyParam2DAsync__funptr)(pCopy,stream)


cdef void* _hipMemcpy2DAsync__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   dst    Destination memory address
#  @param[in]   dpitch Pitch of destination memory
#  @param[in]   src    Source memory address
#  @param[in]   spitch Pitch of source memory
#  @param[in]   width  Width of matrix transfer (columns in bytes)
#  @param[in]   height Height of matrix transfer (rows)
#  @param[in]   kind   Type of transfer
#  @param[in]   stream Stream to use
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DAsync(void * dst,unsigned long dpitch,const void * src,unsigned long spitch,unsigned long width,unsigned long height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpy2DAsync__funptr
    __init_symbol(&_hipMemcpy2DAsync__funptr,"hipMemcpy2DAsync")
    return (<hipError_t (*)(void *,unsigned long,const void *,unsigned long,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpy2DAsync__funptr)(dst,dpitch,src,spitch,width,height,kind,stream)


cdef void* _hipMemcpy2DToArray__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   dst     Destination memory address
#  @param[in]   wOffset Destination starting X offset
#  @param[in]   hOffset Destination starting Y offset
#  @param[in]   src     Source memory address
#  @param[in]   spitch  Pitch of source memory
#  @param[in]   width   Width of matrix transfer (columns in bytes)
#  @param[in]   height  Height of matrix transfer (rows)
#  @param[in]   kind    Type of transfer
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DToArray(hipArray * dst,unsigned long wOffset,unsigned long hOffset,const void * src,unsigned long spitch,unsigned long width,unsigned long height,hipMemcpyKind kind) nogil:
    global _hipMemcpy2DToArray__funptr
    __init_symbol(&_hipMemcpy2DToArray__funptr,"hipMemcpy2DToArray")
    return (<hipError_t (*)(hipArray *,unsigned long,unsigned long,const void *,unsigned long,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpy2DToArray__funptr)(dst,wOffset,hOffset,src,spitch,width,height,kind)


cdef void* _hipMemcpy2DToArrayAsync__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   dst     Destination memory address
#  @param[in]   wOffset Destination starting X offset
#  @param[in]   hOffset Destination starting Y offset
#  @param[in]   src     Source memory address
#  @param[in]   spitch  Pitch of source memory
#  @param[in]   width   Width of matrix transfer (columns in bytes)
#  @param[in]   height  Height of matrix transfer (rows)
#  @param[in]   kind    Type of transfer
#  @param[in]   stream    Accelerator view which the copy is being enqueued
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DToArrayAsync(hipArray * dst,unsigned long wOffset,unsigned long hOffset,const void * src,unsigned long spitch,unsigned long width,unsigned long height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpy2DToArrayAsync__funptr
    __init_symbol(&_hipMemcpy2DToArrayAsync__funptr,"hipMemcpy2DToArrayAsync")
    return (<hipError_t (*)(hipArray *,unsigned long,unsigned long,const void *,unsigned long,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpy2DToArrayAsync__funptr)(dst,wOffset,hOffset,src,spitch,width,height,kind,stream)


cdef void* _hipMemcpyToArray__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   dst     Destination memory address
#  @param[in]   wOffset Destination starting X offset
#  @param[in]   hOffset Destination starting Y offset
#  @param[in]   src     Source memory address
#  @param[in]   count   size in bytes to copy
#  @param[in]   kind    Type of transfer
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpyToArray(hipArray * dst,unsigned long wOffset,unsigned long hOffset,const void * src,unsigned long count,hipMemcpyKind kind) nogil:
    global _hipMemcpyToArray__funptr
    __init_symbol(&_hipMemcpyToArray__funptr,"hipMemcpyToArray")
    return (<hipError_t (*)(hipArray *,unsigned long,unsigned long,const void *,unsigned long,hipMemcpyKind) nogil> _hipMemcpyToArray__funptr)(dst,wOffset,hOffset,src,count,kind)


cdef void* _hipMemcpyFromArray__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   dst       Destination memory address
#  @param[in]   srcArray  Source memory address
#  @param[in]   woffset   Source starting X offset
#  @param[in]   hOffset   Source starting Y offset
#  @param[in]   count     Size in bytes to copy
#  @param[in]   kind      Type of transfer
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpyFromArray(void * dst,hipArray_const_t srcArray,unsigned long wOffset,unsigned long hOffset,unsigned long count,hipMemcpyKind kind) nogil:
    global _hipMemcpyFromArray__funptr
    __init_symbol(&_hipMemcpyFromArray__funptr,"hipMemcpyFromArray")
    return (<hipError_t (*)(void *,hipArray_const_t,unsigned long,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpyFromArray__funptr)(dst,srcArray,wOffset,hOffset,count,kind)


cdef void* _hipMemcpy2DFromArray__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   dst       Destination memory address
#  @param[in]   dpitch    Pitch of destination memory
#  @param[in]   src       Source memory address
#  @param[in]   wOffset   Source starting X offset
#  @param[in]   hOffset   Source starting Y offset
#  @param[in]   width     Width of matrix transfer (columns in bytes)
#  @param[in]   height    Height of matrix transfer (rows)
#  @param[in]   kind      Type of transfer
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DFromArray(void * dst,unsigned long dpitch,hipArray_const_t src,unsigned long wOffset,unsigned long hOffset,unsigned long width,unsigned long height,hipMemcpyKind kind) nogil:
    global _hipMemcpy2DFromArray__funptr
    __init_symbol(&_hipMemcpy2DFromArray__funptr,"hipMemcpy2DFromArray")
    return (<hipError_t (*)(void *,unsigned long,hipArray_const_t,unsigned long,unsigned long,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpy2DFromArray__funptr)(dst,dpitch,src,wOffset,hOffset,width,height,kind)


cdef void* _hipMemcpy2DFromArrayAsync__funptr = NULL
# 
#  @brief Copies data between host and device asynchronously.
# 
#  @param[in]   dst       Destination memory address
#  @param[in]   dpitch    Pitch of destination memory
#  @param[in]   src       Source memory address
#  @param[in]   wOffset   Source starting X offset
#  @param[in]   hOffset   Source starting Y offset
#  @param[in]   width     Width of matrix transfer (columns in bytes)
#  @param[in]   height    Height of matrix transfer (rows)
#  @param[in]   kind      Type of transfer
#  @param[in]   stream    Accelerator view which the copy is being enqueued
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy2DFromArrayAsync(void * dst,unsigned long dpitch,hipArray_const_t src,unsigned long wOffset,unsigned long hOffset,unsigned long width,unsigned long height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpy2DFromArrayAsync__funptr
    __init_symbol(&_hipMemcpy2DFromArrayAsync__funptr,"hipMemcpy2DFromArrayAsync")
    return (<hipError_t (*)(void *,unsigned long,hipArray_const_t,unsigned long,unsigned long,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpy2DFromArrayAsync__funptr)(dst,dpitch,src,wOffset,hOffset,width,height,kind,stream)


cdef void* _hipMemcpyAtoH__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   dst       Destination memory address
#  @param[in]   srcArray  Source array
#  @param[in]   srcoffset Offset in bytes of source array
#  @param[in]   count     Size of memory copy in bytes
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpyAtoH(void * dst,hipArray * srcArray,unsigned long srcOffset,unsigned long count) nogil:
    global _hipMemcpyAtoH__funptr
    __init_symbol(&_hipMemcpyAtoH__funptr,"hipMemcpyAtoH")
    return (<hipError_t (*)(void *,hipArray *,unsigned long,unsigned long) nogil> _hipMemcpyAtoH__funptr)(dst,srcArray,srcOffset,count)


cdef void* _hipMemcpyHtoA__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   dstArray   Destination memory address
#  @param[in]   dstOffset  Offset in bytes of destination array
#  @param[in]   srcHost    Source host pointer
#  @param[in]   count      Size of memory copy in bytes
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpyHtoA(hipArray * dstArray,unsigned long dstOffset,const void * srcHost,unsigned long count) nogil:
    global _hipMemcpyHtoA__funptr
    __init_symbol(&_hipMemcpyHtoA__funptr,"hipMemcpyHtoA")
    return (<hipError_t (*)(hipArray *,unsigned long,const void *,unsigned long) nogil> _hipMemcpyHtoA__funptr)(dstArray,dstOffset,srcHost,count)


cdef void* _hipMemcpy3D__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   p   3D memory copy parameters
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy3D(hipMemcpy3DParms * p) nogil:
    global _hipMemcpy3D__funptr
    __init_symbol(&_hipMemcpy3D__funptr,"hipMemcpy3D")
    return (<hipError_t (*)(hipMemcpy3DParms *) nogil> _hipMemcpy3D__funptr)(p)


cdef void* _hipMemcpy3DAsync__funptr = NULL
# 
#  @brief Copies data between host and device asynchronously.
# 
#  @param[in]   p        3D memory copy parameters
#  @param[in]   stream   Stream to use
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
# #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipMemcpy3DAsync(hipMemcpy3DParms * p,hipStream_t stream) nogil:
    global _hipMemcpy3DAsync__funptr
    __init_symbol(&_hipMemcpy3DAsync__funptr,"hipMemcpy3DAsync")
    return (<hipError_t (*)(hipMemcpy3DParms *,hipStream_t) nogil> _hipMemcpy3DAsync__funptr)(p,stream)


cdef void* _hipDrvMemcpy3D__funptr = NULL
# 
#  @brief Copies data between host and device.
# 
#  @param[in]   pCopy   3D memory copy parameters
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
#  #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipDrvMemcpy3D(HIP_MEMCPY3D * pCopy) nogil:
    global _hipDrvMemcpy3D__funptr
    __init_symbol(&_hipDrvMemcpy3D__funptr,"hipDrvMemcpy3D")
    return (<hipError_t (*)(HIP_MEMCPY3D *) nogil> _hipDrvMemcpy3D__funptr)(pCopy)


cdef void* _hipDrvMemcpy3DAsync__funptr = NULL
# 
#  @brief Copies data between host and device asynchronously.
# 
#  @param[in]   pCopy    3D memory copy parameters
#  @param[in]   stream   Stream to use
#  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
#  #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
# 
#  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
# hipMemcpyAsync
cdef hipError_t hipDrvMemcpy3DAsync(HIP_MEMCPY3D * pCopy,hipStream_t stream) nogil:
    global _hipDrvMemcpy3DAsync__funptr
    __init_symbol(&_hipDrvMemcpy3DAsync__funptr,"hipDrvMemcpy3DAsync")
    return (<hipError_t (*)(HIP_MEMCPY3D *,hipStream_t) nogil> _hipDrvMemcpy3DAsync__funptr)(pCopy,stream)


cdef void* _hipDeviceCanAccessPeer__funptr = NULL
# 
#  @}
# 
# 
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#   @defgroup PeerToPeer PeerToPeer Device Memory Access
#   @{
#   @warning PeerToPeer support is experimental.
#   This section describes the PeerToPeer device memory access functions of HIP runtime API.
# 
# 
#  @brief Determine if a device can access a peer's memory.
# 
#  @param [out] canAccessPeer Returns the peer access capability (0 or 1)
#  @param [in] device - device from where memory may be accessed.
#  @param [in] peerDevice - device where memory is physically located
# 
#  Returns "1" in @p canAccessPeer if the specified @p device is capable
#  of directly accessing memory physically located on peerDevice , or "0" if not.
# 
#  Returns "0" in @p canAccessPeer if deviceId == peerDeviceId, and both are valid devices : a
#  device is not a peer of itself.
# 
#  @returns #hipSuccess,
#  @returns #hipErrorInvalidDevice if deviceId or peerDeviceId are not valid devices
cdef hipError_t hipDeviceCanAccessPeer(int * canAccessPeer,int deviceId,int peerDeviceId) nogil:
    global _hipDeviceCanAccessPeer__funptr
    __init_symbol(&_hipDeviceCanAccessPeer__funptr,"hipDeviceCanAccessPeer")
    return (<hipError_t (*)(int *,int,int) nogil> _hipDeviceCanAccessPeer__funptr)(canAccessPeer,deviceId,peerDeviceId)


cdef void* _hipDeviceEnablePeerAccess__funptr = NULL
# 
# @brief Enable direct access from current device's virtual address space to memory allocations
# physically located on a peer device.
# 
# Memory which already allocated on peer device will be mapped into the address space of the
# current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
# the address space of the current device when the memory is allocated. The peer memory remains
# accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
# 
# 
# @param [in] peerDeviceId
# @param [in] flags
# 
# Returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
# @returns #hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device.
cdef hipError_t hipDeviceEnablePeerAccess(int peerDeviceId,unsigned int flags) nogil:
    global _hipDeviceEnablePeerAccess__funptr
    __init_symbol(&_hipDeviceEnablePeerAccess__funptr,"hipDeviceEnablePeerAccess")
    return (<hipError_t (*)(int,unsigned int) nogil> _hipDeviceEnablePeerAccess__funptr)(peerDeviceId,flags)


cdef void* _hipDeviceDisablePeerAccess__funptr = NULL
# 
# @brief Disable direct access from current device's virtual address space to memory allocations
# physically located on a peer device.
# 
# Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
# enabled from the current device.
# 
# @param [in] peerDeviceId
# 
# @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
cdef hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) nogil:
    global _hipDeviceDisablePeerAccess__funptr
    __init_symbol(&_hipDeviceDisablePeerAccess__funptr,"hipDeviceDisablePeerAccess")
    return (<hipError_t (*)(int) nogil> _hipDeviceDisablePeerAccess__funptr)(peerDeviceId)


cdef void* _hipMemGetAddressRange__funptr = NULL
# 
# @brief Get information on memory allocations.
# 
# @param [out] pbase - BAse pointer address
# @param [out] psize - Size of allocation
# @param [in]  dptr- Device Pointer
# 
# @returns #hipSuccess, #hipErrorInvalidDevicePointer
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipMemGetAddressRange(void ** pbase,unsigned long * psize,void * dptr) nogil:
    global _hipMemGetAddressRange__funptr
    __init_symbol(&_hipMemGetAddressRange__funptr,"hipMemGetAddressRange")
    return (<hipError_t (*)(void **,unsigned long *,void *) nogil> _hipMemGetAddressRange__funptr)(pbase,psize,dptr)


cdef void* _hipMemcpyPeer__funptr = NULL
# 
# @brief Copies memory from one device to memory on another device.
# 
# @param [out] dst - Destination device pointer.
# @param [in] dstDeviceId - Destination device
# @param [in] src - Source device pointer
# @param [in] srcDeviceId - Source device
# @param [in] sizeBytes - Size of memory copy in bytes
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
cdef hipError_t hipMemcpyPeer(void * dst,int dstDeviceId,const void * src,int srcDeviceId,unsigned long sizeBytes) nogil:
    global _hipMemcpyPeer__funptr
    __init_symbol(&_hipMemcpyPeer__funptr,"hipMemcpyPeer")
    return (<hipError_t (*)(void *,int,const void *,int,unsigned long) nogil> _hipMemcpyPeer__funptr)(dst,dstDeviceId,src,srcDeviceId,sizeBytes)


cdef void* _hipMemcpyPeerAsync__funptr = NULL
# 
# @brief Copies memory from one device to memory on another device.
# 
# @param [out] dst - Destination device pointer.
# @param [in] dstDevice - Destination device
# @param [in] src - Source device pointer
# @param [in] srcDevice - Source device
# @param [in] sizeBytes - Size of memory copy in bytes
# @param [in] stream - Stream identifier
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
cdef hipError_t hipMemcpyPeerAsync(void * dst,int dstDeviceId,const void * src,int srcDevice,unsigned long sizeBytes,hipStream_t stream) nogil:
    global _hipMemcpyPeerAsync__funptr
    __init_symbol(&_hipMemcpyPeerAsync__funptr,"hipMemcpyPeerAsync")
    return (<hipError_t (*)(void *,int,const void *,int,unsigned long,hipStream_t) nogil> _hipMemcpyPeerAsync__funptr)(dst,dstDeviceId,src,srcDevice,sizeBytes,stream)


cdef void* _hipCtxCreate__funptr = NULL
# 
#  @}
# 
# 
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#   @defgroup Context Context Management
#   @{
#   This section describes the context management functions of HIP runtime API.
# 
# 
# 
#   @addtogroup ContextD Context Management [Deprecated]
#   @{
#   @ingroup Context
#   This section describes the deprecated context management functions of HIP runtime API.
# 
# 
#  @brief Create a context and set it as current/ default context
# 
#  @param [out] ctx
#  @param [in] flags
#  @param [in] associated device handle
# 
#  @return #hipSuccess
# 
#  @see hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent,
#  hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxCreate(hipCtx_t* ctx,unsigned int flags,int device) nogil:
    global _hipCtxCreate__funptr
    __init_symbol(&_hipCtxCreate__funptr,"hipCtxCreate")
    return (<hipError_t (*)(hipCtx_t*,unsigned int,int) nogil> _hipCtxCreate__funptr)(ctx,flags,device)


cdef void* _hipCtxDestroy__funptr = NULL
# 
# @brief Destroy a HIP context.
# 
# @param [in] ctx Context to destroy
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @see hipCtxCreate, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,hipCtxSetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
cdef hipError_t hipCtxDestroy(hipCtx_t ctx) nogil:
    global _hipCtxDestroy__funptr
    __init_symbol(&_hipCtxDestroy__funptr,"hipCtxDestroy")
    return (<hipError_t (*)(hipCtx_t) nogil> _hipCtxDestroy__funptr)(ctx)


cdef void* _hipCtxPopCurrent__funptr = NULL
# 
# @brief Pop the current/default context and return the popped context.
# 
# @param [out] ctx
# 
# @returns #hipSuccess, #hipErrorInvalidContext
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxSetCurrent, hipCtxGetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxPopCurrent(hipCtx_t* ctx) nogil:
    global _hipCtxPopCurrent__funptr
    __init_symbol(&_hipCtxPopCurrent__funptr,"hipCtxPopCurrent")
    return (<hipError_t (*)(hipCtx_t*) nogil> _hipCtxPopCurrent__funptr)(ctx)


cdef void* _hipCtxPushCurrent__funptr = NULL
# 
# @brief Push the context to be set as current/ default context
# 
# @param [in] ctx
# 
# @returns #hipSuccess, #hipErrorInvalidContext
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
cdef hipError_t hipCtxPushCurrent(hipCtx_t ctx) nogil:
    global _hipCtxPushCurrent__funptr
    __init_symbol(&_hipCtxPushCurrent__funptr,"hipCtxPushCurrent")
    return (<hipError_t (*)(hipCtx_t) nogil> _hipCtxPushCurrent__funptr)(ctx)


cdef void* _hipCtxSetCurrent__funptr = NULL
# 
# @brief Set the passed context as current/default
# 
# @param [in] ctx
# 
# @returns #hipSuccess, #hipErrorInvalidContext
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
cdef hipError_t hipCtxSetCurrent(hipCtx_t ctx) nogil:
    global _hipCtxSetCurrent__funptr
    __init_symbol(&_hipCtxSetCurrent__funptr,"hipCtxSetCurrent")
    return (<hipError_t (*)(hipCtx_t) nogil> _hipCtxSetCurrent__funptr)(ctx)


cdef void* _hipCtxGetCurrent__funptr = NULL
# 
# @brief Get the handle of the current/ default context
# 
# @param [out] ctx
# 
# @returns #hipSuccess, #hipErrorInvalidContext
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetCurrent(hipCtx_t* ctx) nogil:
    global _hipCtxGetCurrent__funptr
    __init_symbol(&_hipCtxGetCurrent__funptr,"hipCtxGetCurrent")
    return (<hipError_t (*)(hipCtx_t*) nogil> _hipCtxGetCurrent__funptr)(ctx)


cdef void* _hipCtxGetDevice__funptr = NULL
# 
# @brief Get the handle of the device associated with current/default context
# 
# @param [out] device
# 
# @returns #hipSuccess, #hipErrorInvalidContext
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize
cdef hipError_t hipCtxGetDevice(int * device) nogil:
    global _hipCtxGetDevice__funptr
    __init_symbol(&_hipCtxGetDevice__funptr,"hipCtxGetDevice")
    return (<hipError_t (*)(int *) nogil> _hipCtxGetDevice__funptr)(device)


cdef void* _hipCtxGetApiVersion__funptr = NULL
# 
# @brief Returns the approximate HIP api version.
# 
# @param [in]  ctx Context to check
# @param [out] apiVersion
# 
# @return #hipSuccess
# 
# @warning The HIP feature set does not correspond to an exact CUDA SDK api revision.
# This function always set *apiVersion to 4 as an approximation though HIP supports
# some features which were introduced in later CUDA SDK revisions.
# HIP apps code should not rely on the api revision number here and should
# use arch feature flags to test device capabilities or conditional compilation.
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
# hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetApiVersion(hipCtx_t ctx,int * apiVersion) nogil:
    global _hipCtxGetApiVersion__funptr
    __init_symbol(&_hipCtxGetApiVersion__funptr,"hipCtxGetApiVersion")
    return (<hipError_t (*)(hipCtx_t,int *) nogil> _hipCtxGetApiVersion__funptr)(ctx,apiVersion)


cdef void* _hipCtxGetCacheConfig__funptr = NULL
# 
# @brief Set Cache configuration for a specific function
# 
# @param [out] cacheConfiguration
# 
# @return #hipSuccess
# 
# @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
# ignored on those architectures.
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetCacheConfig(hipFuncCache_t * cacheConfig) nogil:
    global _hipCtxGetCacheConfig__funptr
    __init_symbol(&_hipCtxGetCacheConfig__funptr,"hipCtxGetCacheConfig")
    return (<hipError_t (*)(hipFuncCache_t *) nogil> _hipCtxGetCacheConfig__funptr)(cacheConfig)


cdef void* _hipCtxSetCacheConfig__funptr = NULL
# 
# @brief Set L1/Shared cache partition.
# 
# @param [in] cacheConfiguration
# 
# @return #hipSuccess
# 
# @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
# ignored on those architectures.
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) nogil:
    global _hipCtxSetCacheConfig__funptr
    __init_symbol(&_hipCtxSetCacheConfig__funptr,"hipCtxSetCacheConfig")
    return (<hipError_t (*)(hipFuncCache_t) nogil> _hipCtxSetCacheConfig__funptr)(cacheConfig)


cdef void* _hipCtxSetSharedMemConfig__funptr = NULL
# 
# @brief Set Shared memory bank configuration.
# 
# @param [in] sharedMemoryConfiguration
# 
# @return #hipSuccess
# 
# @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) nogil:
    global _hipCtxSetSharedMemConfig__funptr
    __init_symbol(&_hipCtxSetSharedMemConfig__funptr,"hipCtxSetSharedMemConfig")
    return (<hipError_t (*)(hipSharedMemConfig) nogil> _hipCtxSetSharedMemConfig__funptr)(config)


cdef void* _hipCtxGetSharedMemConfig__funptr = NULL
# 
# @brief Get Shared memory bank configuration.
# 
# @param [out] sharedMemoryConfiguration
# 
# @return #hipSuccess
# 
# @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
# ignored on those architectures.
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig * pConfig) nogil:
    global _hipCtxGetSharedMemConfig__funptr
    __init_symbol(&_hipCtxGetSharedMemConfig__funptr,"hipCtxGetSharedMemConfig")
    return (<hipError_t (*)(hipSharedMemConfig *) nogil> _hipCtxGetSharedMemConfig__funptr)(pConfig)


cdef void* _hipCtxSynchronize__funptr = NULL
# 
# @brief Blocks until the default context has completed all preceding requested tasks.
# 
# @return #hipSuccess
# 
# @warning This function waits for all streams on the default context to complete execution, and
# then returns.
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxGetDevice
cdef hipError_t hipCtxSynchronize() nogil:
    global _hipCtxSynchronize__funptr
    __init_symbol(&_hipCtxSynchronize__funptr,"hipCtxSynchronize")
    return (<hipError_t (*)() nogil> _hipCtxSynchronize__funptr)()


cdef void* _hipCtxGetFlags__funptr = NULL
# 
# @brief Return flags used for creating default context.
# 
# @param [out] flags
# 
# @returns #hipSuccess
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipCtxGetFlags(unsigned int * flags) nogil:
    global _hipCtxGetFlags__funptr
    __init_symbol(&_hipCtxGetFlags__funptr,"hipCtxGetFlags")
    return (<hipError_t (*)(unsigned int *) nogil> _hipCtxGetFlags__funptr)(flags)


cdef void* _hipCtxEnablePeerAccess__funptr = NULL
# 
# @brief Enables direct access to memory allocations in a peer context.
# 
# Memory which already allocated on peer device will be mapped into the address space of the
# current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
# the address space of the current device when the memory is allocated. The peer memory remains
# accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
# 
# 
# @param [in] peerCtx
# @param [in] flags
# 
# @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
# #hipErrorPeerAccessAlreadyEnabled
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
# @warning PeerToPeer support is experimental.
cdef hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx,unsigned int flags) nogil:
    global _hipCtxEnablePeerAccess__funptr
    __init_symbol(&_hipCtxEnablePeerAccess__funptr,"hipCtxEnablePeerAccess")
    return (<hipError_t (*)(hipCtx_t,unsigned int) nogil> _hipCtxEnablePeerAccess__funptr)(peerCtx,flags)


cdef void* _hipCtxDisablePeerAccess__funptr = NULL
# 
# @brief Disable direct access from current context's virtual address space to memory allocations
# physically located on a peer context.Disables direct access to memory allocations in a peer
# context and unregisters any registered allocations.
# 
# Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
# enabled from the current device.
# 
# @param [in] peerCtx
# 
# @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
# @warning PeerToPeer support is experimental.
cdef hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) nogil:
    global _hipCtxDisablePeerAccess__funptr
    __init_symbol(&_hipCtxDisablePeerAccess__funptr,"hipCtxDisablePeerAccess")
    return (<hipError_t (*)(hipCtx_t) nogil> _hipCtxDisablePeerAccess__funptr)(peerCtx)


cdef void* _hipDevicePrimaryCtxGetState__funptr = NULL
# 
# @}
# 
# 
# @brief Get the state of the primary context.
# 
# @param [in] Device to get primary context flags for
# @param [out] Pointer to store flags
# @param [out] Pointer to store context state; 0 = inactive, 1 = active
# 
# @returns #hipSuccess
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipDevicePrimaryCtxGetState(int dev,unsigned int * flags,int * active) nogil:
    global _hipDevicePrimaryCtxGetState__funptr
    __init_symbol(&_hipDevicePrimaryCtxGetState__funptr,"hipDevicePrimaryCtxGetState")
    return (<hipError_t (*)(int,unsigned int *,int *) nogil> _hipDevicePrimaryCtxGetState__funptr)(dev,flags,active)


cdef void* _hipDevicePrimaryCtxRelease__funptr = NULL
# 
# @brief Release the primary context on the GPU.
# 
# @param [in] Device which primary context is released
# 
# @returns #hipSuccess
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
# @warning This function return #hipSuccess though doesn't release the primaryCtx by design on
# HIP/HCC path.
cdef hipError_t hipDevicePrimaryCtxRelease(int dev) nogil:
    global _hipDevicePrimaryCtxRelease__funptr
    __init_symbol(&_hipDevicePrimaryCtxRelease__funptr,"hipDevicePrimaryCtxRelease")
    return (<hipError_t (*)(int) nogil> _hipDevicePrimaryCtxRelease__funptr)(dev)


cdef void* _hipDevicePrimaryCtxRetain__funptr = NULL
# 
# @brief Retain the primary context on the GPU.
# 
# @param [out] Returned context handle of the new context
# @param [in] Device which primary context is released
# 
# @returns #hipSuccess
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx,int dev) nogil:
    global _hipDevicePrimaryCtxRetain__funptr
    __init_symbol(&_hipDevicePrimaryCtxRetain__funptr,"hipDevicePrimaryCtxRetain")
    return (<hipError_t (*)(hipCtx_t*,int) nogil> _hipDevicePrimaryCtxRetain__funptr)(pctx,dev)


cdef void* _hipDevicePrimaryCtxReset__funptr = NULL
# 
# @brief Resets the primary context on the GPU.
# 
# @param [in] Device which primary context is reset
# 
# @returns #hipSuccess
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipDevicePrimaryCtxReset(int dev) nogil:
    global _hipDevicePrimaryCtxReset__funptr
    __init_symbol(&_hipDevicePrimaryCtxReset__funptr,"hipDevicePrimaryCtxReset")
    return (<hipError_t (*)(int) nogil> _hipDevicePrimaryCtxReset__funptr)(dev)


cdef void* _hipDevicePrimaryCtxSetFlags__funptr = NULL
# 
# @brief Set flags for the primary context.
# 
# @param [in] Device for which the primary context flags are set
# @param [in] New flags for the device
# 
# @returns #hipSuccess, #hipErrorContextAlreadyInUse
# 
# @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
# hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
cdef hipError_t hipDevicePrimaryCtxSetFlags(int dev,unsigned int flags) nogil:
    global _hipDevicePrimaryCtxSetFlags__funptr
    __init_symbol(&_hipDevicePrimaryCtxSetFlags__funptr,"hipDevicePrimaryCtxSetFlags")
    return (<hipError_t (*)(int,unsigned int) nogil> _hipDevicePrimaryCtxSetFlags__funptr)(dev,flags)


cdef void* _hipModuleLoad__funptr = NULL
# 
# @}
# 
# 
# 
#  @defgroup Module Module Management
#  @{
#  This section describes the module management functions of HIP runtime API.
# 
# 
# 
# @brief Loads code object from file into a hipModule_t
# 
# @param [in] fname
# @param [out] module
# 
# @warning File/memory resources allocated in this function are released only in hipModuleUnload.
# 
# @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorFileNotFound,
# hipErrorOutOfMemory, hipErrorSharedObjectInitFailed, hipErrorNotInitialized
# 
#
cdef hipError_t hipModuleLoad(hipModule_t* module,const char * fname) nogil:
    global _hipModuleLoad__funptr
    __init_symbol(&_hipModuleLoad__funptr,"hipModuleLoad")
    return (<hipError_t (*)(hipModule_t*,const char *) nogil> _hipModuleLoad__funptr)(module,fname)


cdef void* _hipModuleUnload__funptr = NULL
# 
# @brief Frees the module
# 
# @param [in] module
# 
# @returns hipSuccess, hipInvalidValue
# module is freed and the code objects associated with it are destroyed
#
cdef hipError_t hipModuleUnload(hipModule_t module) nogil:
    global _hipModuleUnload__funptr
    __init_symbol(&_hipModuleUnload__funptr,"hipModuleUnload")
    return (<hipError_t (*)(hipModule_t) nogil> _hipModuleUnload__funptr)(module)


cdef void* _hipModuleGetFunction__funptr = NULL
# 
# @brief Function with kname will be extracted if present in module
# 
# @param [in] module
# @param [in] kname
# @param [out] function
# 
# @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorNotInitialized,
# hipErrorNotFound,
cdef hipError_t hipModuleGetFunction(hipFunction_t* function,hipModule_t module,const char * kname) nogil:
    global _hipModuleGetFunction__funptr
    __init_symbol(&_hipModuleGetFunction__funptr,"hipModuleGetFunction")
    return (<hipError_t (*)(hipFunction_t*,hipModule_t,const char *) nogil> _hipModuleGetFunction__funptr)(function,module,kname)


cdef void* _hipFuncGetAttributes__funptr = NULL
# 
# @brief Find out attributes for a given function.
# 
# @param [out] attr
# @param [in] func
# 
# @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
cdef hipError_t hipFuncGetAttributes(hipFuncAttributes * attr,const void * func) nogil:
    global _hipFuncGetAttributes__funptr
    __init_symbol(&_hipFuncGetAttributes__funptr,"hipFuncGetAttributes")
    return (<hipError_t (*)(hipFuncAttributes *,const void *) nogil> _hipFuncGetAttributes__funptr)(attr,func)


cdef void* _hipFuncGetAttribute__funptr = NULL
# 
# @brief Find out a specific attribute for a given function.
# 
# @param [out] value
# @param [in]  attrib
# @param [in]  hfunc
# 
# @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
cdef hipError_t hipFuncGetAttribute(int * value,hipFunction_attribute attrib,hipFunction_t hfunc) nogil:
    global _hipFuncGetAttribute__funptr
    __init_symbol(&_hipFuncGetAttribute__funptr,"hipFuncGetAttribute")
    return (<hipError_t (*)(int *,hipFunction_attribute,hipFunction_t) nogil> _hipFuncGetAttribute__funptr)(value,attrib,hfunc)


cdef void* _hipModuleGetTexRef__funptr = NULL
# 
# @brief returns the handle of the texture reference with the name from the module.
# 
# @param [in] hmod
# @param [in] name
# @param [out] texRef
# 
# @returns hipSuccess, hipErrorNotInitialized, hipErrorNotFound, hipErrorInvalidValue
cdef hipError_t hipModuleGetTexRef(textureReference ** texRef,hipModule_t hmod,const char * name) nogil:
    global _hipModuleGetTexRef__funptr
    __init_symbol(&_hipModuleGetTexRef__funptr,"hipModuleGetTexRef")
    return (<hipError_t (*)(textureReference **,hipModule_t,const char *) nogil> _hipModuleGetTexRef__funptr)(texRef,hmod,name)


cdef void* _hipModuleLoadData__funptr = NULL
# 
# @brief builds module from code object which resides in host memory. Image is pointer to that
# location.
# 
# @param [in] image
# @param [out] module
# 
# @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
cdef hipError_t hipModuleLoadData(hipModule_t* module,const void * image) nogil:
    global _hipModuleLoadData__funptr
    __init_symbol(&_hipModuleLoadData__funptr,"hipModuleLoadData")
    return (<hipError_t (*)(hipModule_t*,const void *) nogil> _hipModuleLoadData__funptr)(module,image)


cdef void* _hipModuleLoadDataEx__funptr = NULL
# 
# @brief builds module from code object which resides in host memory. Image is pointer to that
# location. Options are not used. hipModuleLoadData is called.
# 
# @param [in] image
# @param [out] module
# @param [in] number of options
# @param [in] options for JIT
# @param [in] option values for JIT
# 
# @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
cdef hipError_t hipModuleLoadDataEx(hipModule_t* module,const void * image,unsigned int numOptions,hipJitOption * options,void ** optionValues) nogil:
    global _hipModuleLoadDataEx__funptr
    __init_symbol(&_hipModuleLoadDataEx__funptr,"hipModuleLoadDataEx")
    return (<hipError_t (*)(hipModule_t*,const void *,unsigned int,hipJitOption *,void **) nogil> _hipModuleLoadDataEx__funptr)(module,image,numOptions,options,optionValues)


cdef void* _hipModuleLaunchKernel__funptr = NULL
# 
# @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
# to kernelparams or extra
# 
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
# 
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32. So gridDim.x * blockDim.x, gridDim.y * blockDim.y
# and gridDim.z * blockDim.z are always less than 2^32.
# 
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
# 
# @warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please
# refer to hip_porting_driver_api.md for sample usage.
cdef hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,hipStream_t stream,void ** kernelParams,void ** extra) nogil:
    global _hipModuleLaunchKernel__funptr
    __init_symbol(&_hipModuleLaunchKernel__funptr,"hipModuleLaunchKernel")
    return (<hipError_t (*)(hipFunction_t,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,hipStream_t,void **,void **) nogil> _hipModuleLaunchKernel__funptr)(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,stream,kernelParams,extra)


cdef void* _hipLaunchCooperativeKernel__funptr = NULL
# 
# @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
# to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute
# 
# @param [in] f         Kernel to launch.
# @param [in] gridDim   Grid dimensions specified as multiple of blockDim.
# @param [in] blockDim  Block dimensions specified in work-items
# @param [in] kernelParams A list of kernel arguments
# @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
# HIP-Clang compiler provides support for extern shared declarations.
# @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
# default stream is used with associated synchronization rules.
# 
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32.
# 
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
cdef hipError_t hipLaunchCooperativeKernel(const void * f,dim3 gridDim,dim3 blockDimX,void ** kernelParams,unsigned int sharedMemBytes,hipStream_t stream) nogil:
    global _hipLaunchCooperativeKernel__funptr
    __init_symbol(&_hipLaunchCooperativeKernel__funptr,"hipLaunchCooperativeKernel")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,unsigned int,hipStream_t) nogil> _hipLaunchCooperativeKernel__funptr)(f,gridDim,blockDimX,kernelParams,sharedMemBytes,stream)


cdef void* _hipLaunchCooperativeKernelMultiDevice__funptr = NULL
# 
# @brief Launches kernels on multiple devices where thread blocks can cooperate and
# synchronize as they execute.
# 
# @param [in] launchParamsList         List of launch parameters, one per device.
# @param [in] numDevices               Size of the launchParamsList array.
# @param [in] flags                    Flags to control launch behavior.
# 
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
cdef hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams_t * launchParamsList,int numDevices,unsigned int flags) nogil:
    global _hipLaunchCooperativeKernelMultiDevice__funptr
    __init_symbol(&_hipLaunchCooperativeKernelMultiDevice__funptr,"hipLaunchCooperativeKernelMultiDevice")
    return (<hipError_t (*)(hipLaunchParams_t *,int,unsigned int) nogil> _hipLaunchCooperativeKernelMultiDevice__funptr)(launchParamsList,numDevices,flags)


cdef void* _hipExtLaunchMultiKernelMultiDevice__funptr = NULL
# 
# @brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
# on respective streams before enqueuing any other work on the specified streams from any other threads
# 
# 
# @param [in] hipLaunchParams          List of launch parameters, one per device.
# @param [in] numDevices               Size of the launchParamsList array.
# @param [in] flags                    Flags to control launch behavior.
# 
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
cdef hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams_t * launchParamsList,int numDevices,unsigned int flags) nogil:
    global _hipExtLaunchMultiKernelMultiDevice__funptr
    __init_symbol(&_hipExtLaunchMultiKernelMultiDevice__funptr,"hipExtLaunchMultiKernelMultiDevice")
    return (<hipError_t (*)(hipLaunchParams_t *,int,unsigned int) nogil> _hipExtLaunchMultiKernelMultiDevice__funptr)(launchParamsList,numDevices,flags)


cdef void* _hipModuleOccupancyMaxPotentialBlockSize__funptr = NULL
# 
# @}
# 
# 
# 
#  @defgroup Occupancy Occupancy
#  @{
#  This section describes the occupancy functions of HIP runtime API.
# 
# 
# 
# @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
# 
# @param [out] gridSize           minimum grid size for maximum potential occupancy
# @param [out] blockSize          block size for maximum potential occupancy
# @param [in]  f                  kernel function for which occupancy is calulated
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
# 
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32.
# 
# @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
cdef hipError_t hipModuleOccupancyMaxPotentialBlockSize(int * gridSize,int * blockSize,hipFunction_t f,unsigned long dynSharedMemPerBlk,int blockSizeLimit) nogil:
    global _hipModuleOccupancyMaxPotentialBlockSize__funptr
    __init_symbol(&_hipModuleOccupancyMaxPotentialBlockSize__funptr,"hipModuleOccupancyMaxPotentialBlockSize")
    return (<hipError_t (*)(int *,int *,hipFunction_t,unsigned long,int) nogil> _hipModuleOccupancyMaxPotentialBlockSize__funptr)(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit)


cdef void* _hipModuleOccupancyMaxPotentialBlockSizeWithFlags__funptr = NULL
# 
# @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
# 
# @param [out] gridSize           minimum grid size for maximum potential occupancy
# @param [out] blockSize          block size for maximum potential occupancy
# @param [in]  f                  kernel function for which occupancy is calulated
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
# @param [in]  flags            Extra flags for occupancy calculation (only default supported)
# 
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32.
# 
# @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
cdef hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int * gridSize,int * blockSize,hipFunction_t f,unsigned long dynSharedMemPerBlk,int blockSizeLimit,unsigned int flags) nogil:
    global _hipModuleOccupancyMaxPotentialBlockSizeWithFlags__funptr
    __init_symbol(&_hipModuleOccupancyMaxPotentialBlockSizeWithFlags__funptr,"hipModuleOccupancyMaxPotentialBlockSizeWithFlags")
    return (<hipError_t (*)(int *,int *,hipFunction_t,unsigned long,int,unsigned int) nogil> _hipModuleOccupancyMaxPotentialBlockSizeWithFlags__funptr)(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit,flags)


cdef void* _hipModuleOccupancyMaxActiveBlocksPerMultiprocessor__funptr = NULL
# 
# @brief Returns occupancy for a device function.
# 
# @param [out] numBlocks        Returned occupancy
# @param [in]  func             Kernel function (hipFunction) for which occupancy is calulated
# @param [in]  blockSize        Block size the kernel is intended to be launched with
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
cdef hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,hipFunction_t f,int blockSize,unsigned long dynSharedMemPerBlk) nogil:
    global _hipModuleOccupancyMaxActiveBlocksPerMultiprocessor__funptr
    __init_symbol(&_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor__funptr,"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor")
    return (<hipError_t (*)(int *,hipFunction_t,int,unsigned long) nogil> _hipModuleOccupancyMaxActiveBlocksPerMultiprocessor__funptr)(numBlocks,f,blockSize,dynSharedMemPerBlk)


cdef void* _hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags__funptr = NULL
# 
# @brief Returns occupancy for a device function.
# 
# @param [out] numBlocks        Returned occupancy
# @param [in]  f                Kernel function(hipFunction_t) for which occupancy is calulated
# @param [in]  blockSize        Block size the kernel is intended to be launched with
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  flags            Extra flags for occupancy calculation (only default supported)
cdef hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,hipFunction_t f,int blockSize,unsigned long dynSharedMemPerBlk,unsigned int flags) nogil:
    global _hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags__funptr
    __init_symbol(&_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags__funptr,"hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")
    return (<hipError_t (*)(int *,hipFunction_t,int,unsigned long,unsigned int) nogil> _hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags__funptr)(numBlocks,f,blockSize,dynSharedMemPerBlk,flags)


cdef void* _hipOccupancyMaxActiveBlocksPerMultiprocessor__funptr = NULL
# 
# @brief Returns occupancy for a device function.
# 
# @param [out] numBlocks        Returned occupancy
# @param [in]  func             Kernel function for which occupancy is calulated
# @param [in]  blockSize        Block size the kernel is intended to be launched with
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
cdef hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,const void * f,int blockSize,unsigned long dynSharedMemPerBlk) nogil:
    global _hipOccupancyMaxActiveBlocksPerMultiprocessor__funptr
    __init_symbol(&_hipOccupancyMaxActiveBlocksPerMultiprocessor__funptr,"hipOccupancyMaxActiveBlocksPerMultiprocessor")
    return (<hipError_t (*)(int *,const void *,int,unsigned long) nogil> _hipOccupancyMaxActiveBlocksPerMultiprocessor__funptr)(numBlocks,f,blockSize,dynSharedMemPerBlk)


cdef void* _hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags__funptr = NULL
# 
# @brief Returns occupancy for a device function.
# 
# @param [out] numBlocks        Returned occupancy
# @param [in]  f                Kernel function for which occupancy is calulated
# @param [in]  blockSize        Block size the kernel is intended to be launched with
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  flags            Extra flags for occupancy calculation (currently ignored)
cdef hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,const void * f,int blockSize,unsigned long dynSharedMemPerBlk,unsigned int flags) nogil:
    global _hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags__funptr
    __init_symbol(&_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags__funptr,"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")
    return (<hipError_t (*)(int *,const void *,int,unsigned long,unsigned int) nogil> _hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags__funptr)(numBlocks,f,blockSize,dynSharedMemPerBlk,flags)


cdef void* _hipOccupancyMaxPotentialBlockSize__funptr = NULL
# 
# @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
# 
# @param [out] gridSize           minimum grid size for maximum potential occupancy
# @param [out] blockSize          block size for maximum potential occupancy
# @param [in]  f                  kernel function for which occupancy is calulated
# @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
# @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
# 
# Please note, HIP does not support kernel launch with total work items defined in dimension with
# size gridDim x blockDim >= 2^32.
# 
# @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
cdef hipError_t hipOccupancyMaxPotentialBlockSize(int * gridSize,int * blockSize,const void * f,unsigned long dynSharedMemPerBlk,int blockSizeLimit) nogil:
    global _hipOccupancyMaxPotentialBlockSize__funptr
    __init_symbol(&_hipOccupancyMaxPotentialBlockSize__funptr,"hipOccupancyMaxPotentialBlockSize")
    return (<hipError_t (*)(int *,int *,const void *,unsigned long,int) nogil> _hipOccupancyMaxPotentialBlockSize__funptr)(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit)


cdef void* _hipProfilerStart__funptr = NULL
# 
# @brief Start recording of profiling information
# When using this API, start the profiler with profiling disabled.  (--startdisabled)
# @warning : hipProfilerStart API is under development.
cdef hipError_t hipProfilerStart() nogil:
    global _hipProfilerStart__funptr
    __init_symbol(&_hipProfilerStart__funptr,"hipProfilerStart")
    return (<hipError_t (*)() nogil> _hipProfilerStart__funptr)()


cdef void* _hipProfilerStop__funptr = NULL
# 
# @brief Stop recording of profiling information.
# When using this API, start the profiler with profiling disabled.  (--startdisabled)
# @warning : hipProfilerStop API is under development.
cdef hipError_t hipProfilerStop() nogil:
    global _hipProfilerStop__funptr
    __init_symbol(&_hipProfilerStop__funptr,"hipProfilerStop")
    return (<hipError_t (*)() nogil> _hipProfilerStop__funptr)()


cdef void* _hipConfigureCall__funptr = NULL
# 
#  @}
# 
# 
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#   @defgroup Clang Launch API to support the triple-chevron syntax
#   @{
#   This section describes the API to support the triple-chevron syntax.
# 
# 
#  @brief Configure a kernel launch.
# 
#  @param [in] gridDim   grid dimension specified as multiple of blockDim.
#  @param [in] blockDim  block dimensions specified in work-items
#  @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
#  HIP-Clang compiler provides support for extern shared declarations.
#  @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
#  default stream is used with associated synchronization rules.
# 
#  Please note, HIP does not support kernel launch with total work items defined in dimension with
#  size gridDim x blockDim >= 2^32.
# 
#  @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
#
cdef hipError_t hipConfigureCall(dim3 gridDim,dim3 blockDim,unsigned long sharedMem,hipStream_t stream) nogil:
    global _hipConfigureCall__funptr
    __init_symbol(&_hipConfigureCall__funptr,"hipConfigureCall")
    return (<hipError_t (*)(dim3,dim3,unsigned long,hipStream_t) nogil> _hipConfigureCall__funptr)(gridDim,blockDim,sharedMem,stream)


cdef void* _hipSetupArgument__funptr = NULL
# 
# @brief Set a kernel argument.
# 
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
# 
# @param [in] arg    Pointer the argument in host memory.
# @param [in] size   Size of the argument.
# @param [in] offset Offset of the argument on the argument stack.
#
cdef hipError_t hipSetupArgument(const void * arg,unsigned long size,unsigned long offset) nogil:
    global _hipSetupArgument__funptr
    __init_symbol(&_hipSetupArgument__funptr,"hipSetupArgument")
    return (<hipError_t (*)(const void *,unsigned long,unsigned long) nogil> _hipSetupArgument__funptr)(arg,size,offset)


cdef void* _hipLaunchByPtr__funptr = NULL
# 
# @brief Launch a kernel.
# 
# @param [in] func Kernel to launch.
# 
# @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
#
cdef hipError_t hipLaunchByPtr(const void * func) nogil:
    global _hipLaunchByPtr__funptr
    __init_symbol(&_hipLaunchByPtr__funptr,"hipLaunchByPtr")
    return (<hipError_t (*)(const void *) nogil> _hipLaunchByPtr__funptr)(func)


cdef void* _hipLaunchKernel__funptr = NULL
# 
# @brief C compliant kernel launch API
# 
# @param [in] function_address - kernel stub function pointer.
# @param [in] numBlocks - number of blocks
# @param [in] dimBlocks - dimension of a block
# @param [in] args - kernel arguments
# @param [in] sharedMemBytes - Amount of dynamic shared memory to allocate for this kernel. The
# HIP-Clang compiler provides support for extern shared declarations.
# @param [in] stream - Stream where the kernel should be dispatched.  May be 0, in which case th
#  default stream is used with associated synchronization rules.
# 
# @returns #hipSuccess, #hipErrorInvalidValue, hipInvalidDevice
#
cdef hipError_t hipLaunchKernel(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,unsigned long sharedMemBytes,hipStream_t stream) nogil:
    global _hipLaunchKernel__funptr
    __init_symbol(&_hipLaunchKernel__funptr,"hipLaunchKernel")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,unsigned long,hipStream_t) nogil> _hipLaunchKernel__funptr)(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream)


cdef void* _hipLaunchHostFunc__funptr = NULL
# 
# @brief Enqueues a host function call in a stream.
# 
# @param [in] stream - stream to enqueue work to.
# @param [in] fn - function to call once operations enqueued preceeding are complete.
# @param [in] userData - User-specified data to be passed to the function.
# @returns #hipSuccess, #hipErrorInvalidResourceHandle, #hipErrorInvalidValue,
# #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipLaunchHostFunc(hipStream_t stream,hipHostFn_t fn,void * userData):
    global _hipLaunchHostFunc__funptr
    __init_symbol(&_hipLaunchHostFunc__funptr,"hipLaunchHostFunc")
    return (<hipError_t (*)(hipStream_t,hipHostFn_t,void *)> _hipLaunchHostFunc__funptr)(stream,fn,userData)


cdef void* _hipDrvMemcpy2DUnaligned__funptr = NULL
# 
# Copies memory for 2D arrays.
# 
# @param pCopy           - Parameters for the memory copy
# 
# @returns #hipSuccess, #hipErrorInvalidValue
cdef hipError_t hipDrvMemcpy2DUnaligned(hip_Memcpy2D * pCopy) nogil:
    global _hipDrvMemcpy2DUnaligned__funptr
    __init_symbol(&_hipDrvMemcpy2DUnaligned__funptr,"hipDrvMemcpy2DUnaligned")
    return (<hipError_t (*)(hip_Memcpy2D *) nogil> _hipDrvMemcpy2DUnaligned__funptr)(pCopy)


cdef void* _hipExtLaunchKernel__funptr = NULL
# 
# @brief Launches kernel from the pointer address, with arguments and shared memory on stream.
# 
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
#
cdef hipError_t hipExtLaunchKernel(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,unsigned long sharedMemBytes,hipStream_t stream,hipEvent_t startEvent,hipEvent_t stopEvent,int flags) nogil:
    global _hipExtLaunchKernel__funptr
    __init_symbol(&_hipExtLaunchKernel__funptr,"hipExtLaunchKernel")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,unsigned long,hipStream_t,hipEvent_t,hipEvent_t,int) nogil> _hipExtLaunchKernel__funptr)(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream,startEvent,stopEvent,flags)


cdef void* _hipBindTextureToMipmappedArray__funptr = NULL
# 
# @brief  Binds a mipmapped array to a texture.
# 
# @param [in] tex  pointer to the texture reference to bind
# @param [in] mipmappedArray  memory mipmapped array on the device
# @param [in] desc  opointer to the channel format
# 
# @returns hipSuccess, hipErrorInvalidValue
#
cdef hipError_t hipBindTextureToMipmappedArray(textureReference * tex,hipMipmappedArray_const_t mipmappedArray,hipChannelFormatDesc * desc) nogil:
    global _hipBindTextureToMipmappedArray__funptr
    __init_symbol(&_hipBindTextureToMipmappedArray__funptr,"hipBindTextureToMipmappedArray")
    return (<hipError_t (*)(textureReference *,hipMipmappedArray_const_t,hipChannelFormatDesc *) nogil> _hipBindTextureToMipmappedArray__funptr)(tex,mipmappedArray,desc)


cdef void* _hipCreateTextureObject__funptr = NULL
# 
# @brief Creates a texture object.
# 
# @param [out] pTexObject  pointer to the texture object to create
# @param [in] pResDesc  pointer to resource descriptor
# @param [in] pTexDesc  pointer to texture descriptor
# @param [in] pResViewDesc  pointer to resource view descriptor
# 
# @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported, hipErrorOutOfMemory
# 
# @note 3D liner filter isn't supported on GFX90A boards, on which the API @p hipCreateTextureObject will
# return hipErrorNotSupported.
#
cdef hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject,hipResourceDesc * pResDesc,hipTextureDesc * pTexDesc,hipResourceViewDesc * pResViewDesc) nogil:
    global _hipCreateTextureObject__funptr
    __init_symbol(&_hipCreateTextureObject__funptr,"hipCreateTextureObject")
    return (<hipError_t (*)(hipTextureObject_t*,hipResourceDesc *,hipTextureDesc *,hipResourceViewDesc *) nogil> _hipCreateTextureObject__funptr)(pTexObject,pResDesc,pTexDesc,pResViewDesc)


cdef void* _hipDestroyTextureObject__funptr = NULL
# 
# @brief Destroys a texture object.
# 
# @param [in] textureObject  texture object to destroy
# 
# @returns hipSuccess, hipErrorInvalidValue
#
cdef hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) nogil:
    global _hipDestroyTextureObject__funptr
    __init_symbol(&_hipDestroyTextureObject__funptr,"hipDestroyTextureObject")
    return (<hipError_t (*)(hipTextureObject_t) nogil> _hipDestroyTextureObject__funptr)(textureObject)


cdef void* _hipGetChannelDesc__funptr = NULL
# 
# @brief Gets the channel descriptor in an array.
# 
# @param [in] desc  pointer to channel format descriptor
# @param [out] array  memory array on the device
# 
# @returns hipSuccess, hipErrorInvalidValue
#
cdef hipError_t hipGetChannelDesc(hipChannelFormatDesc * desc,hipArray_const_t array) nogil:
    global _hipGetChannelDesc__funptr
    __init_symbol(&_hipGetChannelDesc__funptr,"hipGetChannelDesc")
    return (<hipError_t (*)(hipChannelFormatDesc *,hipArray_const_t) nogil> _hipGetChannelDesc__funptr)(desc,array)


cdef void* _hipGetTextureObjectResourceDesc__funptr = NULL
# 
# @brief Gets resource descriptor for the texture object.
# 
# @param [out] pResDesc  pointer to resource descriptor
# @param [in] textureObject  texture object
# 
# @returns hipSuccess, hipErrorInvalidValue
#
cdef hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc * pResDesc,hipTextureObject_t textureObject) nogil:
    global _hipGetTextureObjectResourceDesc__funptr
    __init_symbol(&_hipGetTextureObjectResourceDesc__funptr,"hipGetTextureObjectResourceDesc")
    return (<hipError_t (*)(hipResourceDesc *,hipTextureObject_t) nogil> _hipGetTextureObjectResourceDesc__funptr)(pResDesc,textureObject)


cdef void* _hipGetTextureObjectResourceViewDesc__funptr = NULL
# 
# @brief Gets resource view descriptor for the texture object.
# 
# @param [out] pResViewDesc  pointer to resource view descriptor
# @param [in] textureObject  texture object
# 
# @returns hipSuccess, hipErrorInvalidValue
#
cdef hipError_t hipGetTextureObjectResourceViewDesc(hipResourceViewDesc * pResViewDesc,hipTextureObject_t textureObject) nogil:
    global _hipGetTextureObjectResourceViewDesc__funptr
    __init_symbol(&_hipGetTextureObjectResourceViewDesc__funptr,"hipGetTextureObjectResourceViewDesc")
    return (<hipError_t (*)(hipResourceViewDesc *,hipTextureObject_t) nogil> _hipGetTextureObjectResourceViewDesc__funptr)(pResViewDesc,textureObject)


cdef void* _hipGetTextureObjectTextureDesc__funptr = NULL
# 
# @brief Gets texture descriptor for the texture object.
# 
# @param [out] pTexDesc  pointer to texture descriptor
# @param [in] textureObject  texture object
# 
# @returns hipSuccess, hipErrorInvalidValue
#
cdef hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc * pTexDesc,hipTextureObject_t textureObject) nogil:
    global _hipGetTextureObjectTextureDesc__funptr
    __init_symbol(&_hipGetTextureObjectTextureDesc__funptr,"hipGetTextureObjectTextureDesc")
    return (<hipError_t (*)(hipTextureDesc *,hipTextureObject_t) nogil> _hipGetTextureObjectTextureDesc__funptr)(pTexDesc,textureObject)


cdef void* _hipTexObjectCreate__funptr = NULL
# 
# @brief Creates a texture object.
# 
# @param [out] pTexObject  pointer to texture object to create
# @param [in] pResDesc  pointer to resource descriptor
# @param [in] pTexDesc  pointer to texture descriptor
# @param [in] pResViewDesc  pointer to resource view descriptor
# 
# @returns hipSuccess, hipErrorInvalidValue
#
cdef hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject,HIP_RESOURCE_DESC_st * pResDesc,HIP_TEXTURE_DESC_st * pTexDesc,HIP_RESOURCE_VIEW_DESC_st * pResViewDesc) nogil:
    global _hipTexObjectCreate__funptr
    __init_symbol(&_hipTexObjectCreate__funptr,"hipTexObjectCreate")
    return (<hipError_t (*)(hipTextureObject_t*,HIP_RESOURCE_DESC_st *,HIP_TEXTURE_DESC_st *,HIP_RESOURCE_VIEW_DESC_st *) nogil> _hipTexObjectCreate__funptr)(pTexObject,pResDesc,pTexDesc,pResViewDesc)


cdef void* _hipTexObjectDestroy__funptr = NULL
# 
# @brief Destroys a texture object.
# 
# @param [in] texObject  texture object to destroy
# 
# @returns hipSuccess, hipErrorInvalidValue
#
cdef hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) nogil:
    global _hipTexObjectDestroy__funptr
    __init_symbol(&_hipTexObjectDestroy__funptr,"hipTexObjectDestroy")
    return (<hipError_t (*)(hipTextureObject_t) nogil> _hipTexObjectDestroy__funptr)(texObject)


cdef void* _hipTexObjectGetResourceDesc__funptr = NULL
# 
# @brief Gets resource descriptor of a texture object.
# 
# @param [out] pResDesc  pointer to resource descriptor
# @param [in] texObject  texture object
# 
# @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
#
cdef hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC_st * pResDesc,hipTextureObject_t texObject) nogil:
    global _hipTexObjectGetResourceDesc__funptr
    __init_symbol(&_hipTexObjectGetResourceDesc__funptr,"hipTexObjectGetResourceDesc")
    return (<hipError_t (*)(HIP_RESOURCE_DESC_st *,hipTextureObject_t) nogil> _hipTexObjectGetResourceDesc__funptr)(pResDesc,texObject)


cdef void* _hipTexObjectGetResourceViewDesc__funptr = NULL
# 
# @brief Gets resource view descriptor of a texture object.
# 
# @param [out] pResViewDesc  pointer to resource view descriptor
# @param [in] texObject  texture object
# 
# @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
#
cdef hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC_st * pResViewDesc,hipTextureObject_t texObject) nogil:
    global _hipTexObjectGetResourceViewDesc__funptr
    __init_symbol(&_hipTexObjectGetResourceViewDesc__funptr,"hipTexObjectGetResourceViewDesc")
    return (<hipError_t (*)(HIP_RESOURCE_VIEW_DESC_st *,hipTextureObject_t) nogil> _hipTexObjectGetResourceViewDesc__funptr)(pResViewDesc,texObject)


cdef void* _hipTexObjectGetTextureDesc__funptr = NULL
# 
# @brief Gets texture descriptor of a texture object.
# 
# @param [out] pTexDesc  pointer to texture descriptor
# @param [in] texObject  texture object
# 
# @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
#
cdef hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC_st * pTexDesc,hipTextureObject_t texObject) nogil:
    global _hipTexObjectGetTextureDesc__funptr
    __init_symbol(&_hipTexObjectGetTextureDesc__funptr,"hipTexObjectGetTextureDesc")
    return (<hipError_t (*)(HIP_TEXTURE_DESC_st *,hipTextureObject_t) nogil> _hipTexObjectGetTextureDesc__funptr)(pTexDesc,texObject)


cdef void* _hipGetTextureReference__funptr = NULL
# 
# 
#  @addtogroup TextureD Texture Management [Deprecated]
#  @{
#  @ingroup Texture
#  This section describes the deprecated texture management functions of HIP runtime API.
# 
# 
# @brief Gets the texture reference related with the symbol.
# 
# @param [out] texref  texture reference
# @param [in] symbol  pointer to the symbol related with the texture for the reference
# 
# @returns hipSuccess, hipErrorInvalidValue
#
cdef hipError_t hipGetTextureReference(textureReference ** texref,const void * symbol) nogil:
    global _hipGetTextureReference__funptr
    __init_symbol(&_hipGetTextureReference__funptr,"hipGetTextureReference")
    return (<hipError_t (*)(textureReference **,const void *) nogil> _hipGetTextureReference__funptr)(texref,symbol)


cdef void* _hipTexRefSetAddressMode__funptr = NULL
cdef hipError_t hipTexRefSetAddressMode(textureReference * texRef,int dim,hipTextureAddressMode am) nogil:
    global _hipTexRefSetAddressMode__funptr
    __init_symbol(&_hipTexRefSetAddressMode__funptr,"hipTexRefSetAddressMode")
    return (<hipError_t (*)(textureReference *,int,hipTextureAddressMode) nogil> _hipTexRefSetAddressMode__funptr)(texRef,dim,am)


cdef void* _hipTexRefSetArray__funptr = NULL
cdef hipError_t hipTexRefSetArray(textureReference * tex,hipArray_const_t array,unsigned int flags) nogil:
    global _hipTexRefSetArray__funptr
    __init_symbol(&_hipTexRefSetArray__funptr,"hipTexRefSetArray")
    return (<hipError_t (*)(textureReference *,hipArray_const_t,unsigned int) nogil> _hipTexRefSetArray__funptr)(tex,array,flags)


cdef void* _hipTexRefSetFilterMode__funptr = NULL
cdef hipError_t hipTexRefSetFilterMode(textureReference * texRef,hipTextureFilterMode fm) nogil:
    global _hipTexRefSetFilterMode__funptr
    __init_symbol(&_hipTexRefSetFilterMode__funptr,"hipTexRefSetFilterMode")
    return (<hipError_t (*)(textureReference *,hipTextureFilterMode) nogil> _hipTexRefSetFilterMode__funptr)(texRef,fm)


cdef void* _hipTexRefSetFlags__funptr = NULL
cdef hipError_t hipTexRefSetFlags(textureReference * texRef,unsigned int Flags) nogil:
    global _hipTexRefSetFlags__funptr
    __init_symbol(&_hipTexRefSetFlags__funptr,"hipTexRefSetFlags")
    return (<hipError_t (*)(textureReference *,unsigned int) nogil> _hipTexRefSetFlags__funptr)(texRef,Flags)


cdef void* _hipTexRefSetFormat__funptr = NULL
cdef hipError_t hipTexRefSetFormat(textureReference * texRef,hipArray_Format fmt,int NumPackedComponents) nogil:
    global _hipTexRefSetFormat__funptr
    __init_symbol(&_hipTexRefSetFormat__funptr,"hipTexRefSetFormat")
    return (<hipError_t (*)(textureReference *,hipArray_Format,int) nogil> _hipTexRefSetFormat__funptr)(texRef,fmt,NumPackedComponents)


cdef void* _hipBindTexture__funptr = NULL
cdef hipError_t hipBindTexture(unsigned long * offset,textureReference * tex,const void * devPtr,hipChannelFormatDesc * desc,unsigned long size) nogil:
    global _hipBindTexture__funptr
    __init_symbol(&_hipBindTexture__funptr,"hipBindTexture")
    return (<hipError_t (*)(unsigned long *,textureReference *,const void *,hipChannelFormatDesc *,unsigned long) nogil> _hipBindTexture__funptr)(offset,tex,devPtr,desc,size)


cdef void* _hipBindTexture2D__funptr = NULL
cdef hipError_t hipBindTexture2D(unsigned long * offset,textureReference * tex,const void * devPtr,hipChannelFormatDesc * desc,unsigned long width,unsigned long height,unsigned long pitch) nogil:
    global _hipBindTexture2D__funptr
    __init_symbol(&_hipBindTexture2D__funptr,"hipBindTexture2D")
    return (<hipError_t (*)(unsigned long *,textureReference *,const void *,hipChannelFormatDesc *,unsigned long,unsigned long,unsigned long) nogil> _hipBindTexture2D__funptr)(offset,tex,devPtr,desc,width,height,pitch)


cdef void* _hipBindTextureToArray__funptr = NULL
cdef hipError_t hipBindTextureToArray(textureReference * tex,hipArray_const_t array,hipChannelFormatDesc * desc) nogil:
    global _hipBindTextureToArray__funptr
    __init_symbol(&_hipBindTextureToArray__funptr,"hipBindTextureToArray")
    return (<hipError_t (*)(textureReference *,hipArray_const_t,hipChannelFormatDesc *) nogil> _hipBindTextureToArray__funptr)(tex,array,desc)


cdef void* _hipGetTextureAlignmentOffset__funptr = NULL
cdef hipError_t hipGetTextureAlignmentOffset(unsigned long * offset,textureReference * texref) nogil:
    global _hipGetTextureAlignmentOffset__funptr
    __init_symbol(&_hipGetTextureAlignmentOffset__funptr,"hipGetTextureAlignmentOffset")
    return (<hipError_t (*)(unsigned long *,textureReference *) nogil> _hipGetTextureAlignmentOffset__funptr)(offset,texref)


cdef void* _hipUnbindTexture__funptr = NULL
cdef hipError_t hipUnbindTexture(textureReference * tex) nogil:
    global _hipUnbindTexture__funptr
    __init_symbol(&_hipUnbindTexture__funptr,"hipUnbindTexture")
    return (<hipError_t (*)(textureReference *) nogil> _hipUnbindTexture__funptr)(tex)


cdef void* _hipTexRefGetAddress__funptr = NULL
cdef hipError_t hipTexRefGetAddress(void ** dev_ptr,textureReference * texRef) nogil:
    global _hipTexRefGetAddress__funptr
    __init_symbol(&_hipTexRefGetAddress__funptr,"hipTexRefGetAddress")
    return (<hipError_t (*)(void **,textureReference *) nogil> _hipTexRefGetAddress__funptr)(dev_ptr,texRef)


cdef void* _hipTexRefGetAddressMode__funptr = NULL
cdef hipError_t hipTexRefGetAddressMode(hipTextureAddressMode * pam,textureReference * texRef,int dim) nogil:
    global _hipTexRefGetAddressMode__funptr
    __init_symbol(&_hipTexRefGetAddressMode__funptr,"hipTexRefGetAddressMode")
    return (<hipError_t (*)(hipTextureAddressMode *,textureReference *,int) nogil> _hipTexRefGetAddressMode__funptr)(pam,texRef,dim)


cdef void* _hipTexRefGetFilterMode__funptr = NULL
cdef hipError_t hipTexRefGetFilterMode(hipTextureFilterMode * pfm,textureReference * texRef) nogil:
    global _hipTexRefGetFilterMode__funptr
    __init_symbol(&_hipTexRefGetFilterMode__funptr,"hipTexRefGetFilterMode")
    return (<hipError_t (*)(hipTextureFilterMode *,textureReference *) nogil> _hipTexRefGetFilterMode__funptr)(pfm,texRef)


cdef void* _hipTexRefGetFlags__funptr = NULL
cdef hipError_t hipTexRefGetFlags(unsigned int * pFlags,textureReference * texRef) nogil:
    global _hipTexRefGetFlags__funptr
    __init_symbol(&_hipTexRefGetFlags__funptr,"hipTexRefGetFlags")
    return (<hipError_t (*)(unsigned int *,textureReference *) nogil> _hipTexRefGetFlags__funptr)(pFlags,texRef)


cdef void* _hipTexRefGetFormat__funptr = NULL
cdef hipError_t hipTexRefGetFormat(hipArray_Format * pFormat,int * pNumChannels,textureReference * texRef) nogil:
    global _hipTexRefGetFormat__funptr
    __init_symbol(&_hipTexRefGetFormat__funptr,"hipTexRefGetFormat")
    return (<hipError_t (*)(hipArray_Format *,int *,textureReference *) nogil> _hipTexRefGetFormat__funptr)(pFormat,pNumChannels,texRef)


cdef void* _hipTexRefGetMaxAnisotropy__funptr = NULL
cdef hipError_t hipTexRefGetMaxAnisotropy(int * pmaxAnsio,textureReference * texRef) nogil:
    global _hipTexRefGetMaxAnisotropy__funptr
    __init_symbol(&_hipTexRefGetMaxAnisotropy__funptr,"hipTexRefGetMaxAnisotropy")
    return (<hipError_t (*)(int *,textureReference *) nogil> _hipTexRefGetMaxAnisotropy__funptr)(pmaxAnsio,texRef)


cdef void* _hipTexRefGetMipmapFilterMode__funptr = NULL
cdef hipError_t hipTexRefGetMipmapFilterMode(hipTextureFilterMode * pfm,textureReference * texRef) nogil:
    global _hipTexRefGetMipmapFilterMode__funptr
    __init_symbol(&_hipTexRefGetMipmapFilterMode__funptr,"hipTexRefGetMipmapFilterMode")
    return (<hipError_t (*)(hipTextureFilterMode *,textureReference *) nogil> _hipTexRefGetMipmapFilterMode__funptr)(pfm,texRef)


cdef void* _hipTexRefGetMipmapLevelBias__funptr = NULL
cdef hipError_t hipTexRefGetMipmapLevelBias(float * pbias,textureReference * texRef) nogil:
    global _hipTexRefGetMipmapLevelBias__funptr
    __init_symbol(&_hipTexRefGetMipmapLevelBias__funptr,"hipTexRefGetMipmapLevelBias")
    return (<hipError_t (*)(float *,textureReference *) nogil> _hipTexRefGetMipmapLevelBias__funptr)(pbias,texRef)


cdef void* _hipTexRefGetMipmapLevelClamp__funptr = NULL
cdef hipError_t hipTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp,float * pmaxMipmapLevelClamp,textureReference * texRef) nogil:
    global _hipTexRefGetMipmapLevelClamp__funptr
    __init_symbol(&_hipTexRefGetMipmapLevelClamp__funptr,"hipTexRefGetMipmapLevelClamp")
    return (<hipError_t (*)(float *,float *,textureReference *) nogil> _hipTexRefGetMipmapLevelClamp__funptr)(pminMipmapLevelClamp,pmaxMipmapLevelClamp,texRef)


cdef void* _hipTexRefGetMipMappedArray__funptr = NULL
cdef hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray,textureReference * texRef) nogil:
    global _hipTexRefGetMipMappedArray__funptr
    __init_symbol(&_hipTexRefGetMipMappedArray__funptr,"hipTexRefGetMipMappedArray")
    return (<hipError_t (*)(hipMipmappedArray_t*,textureReference *) nogil> _hipTexRefGetMipMappedArray__funptr)(pArray,texRef)


cdef void* _hipTexRefSetAddress__funptr = NULL
cdef hipError_t hipTexRefSetAddress(unsigned long * ByteOffset,textureReference * texRef,void * dptr,unsigned long bytes) nogil:
    global _hipTexRefSetAddress__funptr
    __init_symbol(&_hipTexRefSetAddress__funptr,"hipTexRefSetAddress")
    return (<hipError_t (*)(unsigned long *,textureReference *,void *,unsigned long) nogil> _hipTexRefSetAddress__funptr)(ByteOffset,texRef,dptr,bytes)


cdef void* _hipTexRefSetAddress2D__funptr = NULL
cdef hipError_t hipTexRefSetAddress2D(textureReference * texRef,HIP_ARRAY_DESCRIPTOR * desc,void * dptr,unsigned long Pitch) nogil:
    global _hipTexRefSetAddress2D__funptr
    __init_symbol(&_hipTexRefSetAddress2D__funptr,"hipTexRefSetAddress2D")
    return (<hipError_t (*)(textureReference *,HIP_ARRAY_DESCRIPTOR *,void *,unsigned long) nogil> _hipTexRefSetAddress2D__funptr)(texRef,desc,dptr,Pitch)


cdef void* _hipTexRefSetMaxAnisotropy__funptr = NULL
cdef hipError_t hipTexRefSetMaxAnisotropy(textureReference * texRef,unsigned int maxAniso) nogil:
    global _hipTexRefSetMaxAnisotropy__funptr
    __init_symbol(&_hipTexRefSetMaxAnisotropy__funptr,"hipTexRefSetMaxAnisotropy")
    return (<hipError_t (*)(textureReference *,unsigned int) nogil> _hipTexRefSetMaxAnisotropy__funptr)(texRef,maxAniso)


cdef void* _hipTexRefSetBorderColor__funptr = NULL
cdef hipError_t hipTexRefSetBorderColor(textureReference * texRef,float * pBorderColor) nogil:
    global _hipTexRefSetBorderColor__funptr
    __init_symbol(&_hipTexRefSetBorderColor__funptr,"hipTexRefSetBorderColor")
    return (<hipError_t (*)(textureReference *,float *) nogil> _hipTexRefSetBorderColor__funptr)(texRef,pBorderColor)


cdef void* _hipTexRefSetMipmapFilterMode__funptr = NULL
cdef hipError_t hipTexRefSetMipmapFilterMode(textureReference * texRef,hipTextureFilterMode fm) nogil:
    global _hipTexRefSetMipmapFilterMode__funptr
    __init_symbol(&_hipTexRefSetMipmapFilterMode__funptr,"hipTexRefSetMipmapFilterMode")
    return (<hipError_t (*)(textureReference *,hipTextureFilterMode) nogil> _hipTexRefSetMipmapFilterMode__funptr)(texRef,fm)


cdef void* _hipTexRefSetMipmapLevelBias__funptr = NULL
cdef hipError_t hipTexRefSetMipmapLevelBias(textureReference * texRef,float bias) nogil:
    global _hipTexRefSetMipmapLevelBias__funptr
    __init_symbol(&_hipTexRefSetMipmapLevelBias__funptr,"hipTexRefSetMipmapLevelBias")
    return (<hipError_t (*)(textureReference *,float) nogil> _hipTexRefSetMipmapLevelBias__funptr)(texRef,bias)


cdef void* _hipTexRefSetMipmapLevelClamp__funptr = NULL
cdef hipError_t hipTexRefSetMipmapLevelClamp(textureReference * texRef,float minMipMapLevelClamp,float maxMipMapLevelClamp) nogil:
    global _hipTexRefSetMipmapLevelClamp__funptr
    __init_symbol(&_hipTexRefSetMipmapLevelClamp__funptr,"hipTexRefSetMipmapLevelClamp")
    return (<hipError_t (*)(textureReference *,float,float) nogil> _hipTexRefSetMipmapLevelClamp__funptr)(texRef,minMipMapLevelClamp,maxMipMapLevelClamp)


cdef void* _hipTexRefSetMipmappedArray__funptr = NULL
cdef hipError_t hipTexRefSetMipmappedArray(textureReference * texRef,hipMipmappedArray * mipmappedArray,unsigned int Flags) nogil:
    global _hipTexRefSetMipmappedArray__funptr
    __init_symbol(&_hipTexRefSetMipmappedArray__funptr,"hipTexRefSetMipmappedArray")
    return (<hipError_t (*)(textureReference *,hipMipmappedArray *,unsigned int) nogil> _hipTexRefSetMipmappedArray__funptr)(texRef,mipmappedArray,Flags)


cdef void* _hipMipmappedArrayCreate__funptr = NULL
# 
# 
# @addtogroup TextureU Texture Management [Not supported]
# @{
# @ingroup Texture
# This section describes the texture management functions currently unsupported in HIP runtime.
cdef hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,HIP_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc,unsigned int numMipmapLevels) nogil:
    global _hipMipmappedArrayCreate__funptr
    __init_symbol(&_hipMipmappedArrayCreate__funptr,"hipMipmappedArrayCreate")
    return (<hipError_t (*)(hipMipmappedArray_t*,HIP_ARRAY3D_DESCRIPTOR *,unsigned int) nogil> _hipMipmappedArrayCreate__funptr)(pHandle,pMipmappedArrayDesc,numMipmapLevels)


cdef void* _hipMipmappedArrayDestroy__funptr = NULL
cdef hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) nogil:
    global _hipMipmappedArrayDestroy__funptr
    __init_symbol(&_hipMipmappedArrayDestroy__funptr,"hipMipmappedArrayDestroy")
    return (<hipError_t (*)(hipMipmappedArray_t) nogil> _hipMipmappedArrayDestroy__funptr)(hMipmappedArray)


cdef void* _hipMipmappedArrayGetLevel__funptr = NULL
cdef hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray,hipMipmappedArray_t hMipMappedArray,unsigned int level) nogil:
    global _hipMipmappedArrayGetLevel__funptr
    __init_symbol(&_hipMipmappedArrayGetLevel__funptr,"hipMipmappedArrayGetLevel")
    return (<hipError_t (*)(hipArray_t*,hipMipmappedArray_t,unsigned int) nogil> _hipMipmappedArrayGetLevel__funptr)(pLevelArray,hMipMappedArray,level)


cdef void* _hipApiName__funptr = NULL
# 
# 
# @defgroup Callback Callback Activity APIs
# @{
# This section describes the callback/Activity of HIP runtime API.
cdef const char * hipApiName(unsigned int id) nogil:
    global _hipApiName__funptr
    __init_symbol(&_hipApiName__funptr,"hipApiName")
    return (<const char * (*)(unsigned int) nogil> _hipApiName__funptr)(id)


cdef void* _hipKernelNameRef__funptr = NULL
cdef const char * hipKernelNameRef(hipFunction_t f) nogil:
    global _hipKernelNameRef__funptr
    __init_symbol(&_hipKernelNameRef__funptr,"hipKernelNameRef")
    return (<const char * (*)(hipFunction_t) nogil> _hipKernelNameRef__funptr)(f)


cdef void* _hipKernelNameRefByPtr__funptr = NULL
cdef const char * hipKernelNameRefByPtr(const void * hostFunction,hipStream_t stream) nogil:
    global _hipKernelNameRefByPtr__funptr
    __init_symbol(&_hipKernelNameRefByPtr__funptr,"hipKernelNameRefByPtr")
    return (<const char * (*)(const void *,hipStream_t) nogil> _hipKernelNameRefByPtr__funptr)(hostFunction,stream)


cdef void* _hipGetStreamDeviceId__funptr = NULL
cdef int hipGetStreamDeviceId(hipStream_t stream) nogil:
    global _hipGetStreamDeviceId__funptr
    __init_symbol(&_hipGetStreamDeviceId__funptr,"hipGetStreamDeviceId")
    return (<int (*)(hipStream_t) nogil> _hipGetStreamDeviceId__funptr)(stream)


cdef void* _hipStreamBeginCapture__funptr = NULL
# 
# @brief Begins graph capture on a stream.
# 
# @param [in] stream - Stream to initiate capture.
# @param [in] mode - Controls the interaction of this capture sequence with other API calls that
# are not safe.
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipStreamBeginCapture(hipStream_t stream,hipStreamCaptureMode mode) nogil:
    global _hipStreamBeginCapture__funptr
    __init_symbol(&_hipStreamBeginCapture__funptr,"hipStreamBeginCapture")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureMode) nogil> _hipStreamBeginCapture__funptr)(stream,mode)


cdef void* _hipStreamEndCapture__funptr = NULL
# 
# @brief Ends capture on a stream, returning the captured graph.
# 
# @param [in] stream - Stream to end capture.
# @param [out] pGraph - returns the graph captured.
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipStreamEndCapture(hipStream_t stream,hipGraph_t* pGraph) nogil:
    global _hipStreamEndCapture__funptr
    __init_symbol(&_hipStreamEndCapture__funptr,"hipStreamEndCapture")
    return (<hipError_t (*)(hipStream_t,hipGraph_t*) nogil> _hipStreamEndCapture__funptr)(stream,pGraph)


cdef void* _hipStreamGetCaptureInfo__funptr = NULL
# 
# @brief Get capture status of a stream.
# 
# @param [in] stream - Stream under capture.
# @param [out] pCaptureStatus - returns current status of the capture.
# @param [out] pId - unique ID of the capture.
# 
# @returns #hipSuccess, #hipErrorStreamCaptureImplicit
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipStreamGetCaptureInfo(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus,unsigned long long * pId) nogil:
    global _hipStreamGetCaptureInfo__funptr
    __init_symbol(&_hipStreamGetCaptureInfo__funptr,"hipStreamGetCaptureInfo")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *) nogil> _hipStreamGetCaptureInfo__funptr)(stream,pCaptureStatus,pId)


cdef void* _hipStreamGetCaptureInfo_v2__funptr = NULL
# 
# @brief Get stream's capture state
# 
# @param [in] stream - Stream under capture.
# @param [out] captureStatus_out - returns current status of the capture.
# @param [out] id_out - unique ID of the capture.
# @param [in] graph_out - returns the graph being captured into.
# @param [out] dependencies_out - returns pointer to an array of nodes.
# @param [out] numDependencies_out - returns size of the array returned in dependencies_out.
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream,hipStreamCaptureStatus * captureStatus_out,unsigned long long * id_out,hipGraph_t* graph_out,hipGraphNode_t ** dependencies_out,unsigned long * numDependencies_out) nogil:
    global _hipStreamGetCaptureInfo_v2__funptr
    __init_symbol(&_hipStreamGetCaptureInfo_v2__funptr,"hipStreamGetCaptureInfo_v2")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *,hipGraph_t*,hipGraphNode_t **,unsigned long *) nogil> _hipStreamGetCaptureInfo_v2__funptr)(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out)


cdef void* _hipStreamIsCapturing__funptr = NULL
# 
# @brief Get stream's capture state
# 
# @param [in] stream - Stream under capture.
# @param [out] pCaptureStatus - returns current status of the capture.
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipStreamIsCapturing(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus) nogil:
    global _hipStreamIsCapturing__funptr
    __init_symbol(&_hipStreamIsCapturing__funptr,"hipStreamIsCapturing")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *) nogil> _hipStreamIsCapturing__funptr)(stream,pCaptureStatus)


cdef void* _hipStreamUpdateCaptureDependencies__funptr = NULL
# 
# @brief Update the set of dependencies in a capturing stream
# 
# @param [in] stream - Stream under capture.
# @param [in] dependencies - pointer to an array of nodes to Add/Replace.
# @param [in] numDependencies - size of the array in dependencies.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorIllegalState
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream,hipGraphNode_t* dependencies,unsigned long numDependencies,unsigned int flags) nogil:
    global _hipStreamUpdateCaptureDependencies__funptr
    __init_symbol(&_hipStreamUpdateCaptureDependencies__funptr,"hipStreamUpdateCaptureDependencies")
    return (<hipError_t (*)(hipStream_t,hipGraphNode_t*,unsigned long,unsigned int) nogil> _hipStreamUpdateCaptureDependencies__funptr)(stream,dependencies,numDependencies,flags)


cdef void* _hipThreadExchangeStreamCaptureMode__funptr = NULL
# 
# @brief Swaps the stream capture mode of a thread.
# 
# @param [in] mode - Pointer to mode value to swap with the current mode
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode * mode) nogil:
    global _hipThreadExchangeStreamCaptureMode__funptr
    __init_symbol(&_hipThreadExchangeStreamCaptureMode__funptr,"hipThreadExchangeStreamCaptureMode")
    return (<hipError_t (*)(hipStreamCaptureMode *) nogil> _hipThreadExchangeStreamCaptureMode__funptr)(mode)


cdef void* _hipGraphCreate__funptr = NULL
# 
# @brief Creates a graph
# 
# @param [out] pGraph - pointer to graph to create.
# @param [in] flags - flags for graph creation, must be 0.
# 
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphCreate(hipGraph_t* pGraph,unsigned int flags) nogil:
    global _hipGraphCreate__funptr
    __init_symbol(&_hipGraphCreate__funptr,"hipGraphCreate")
    return (<hipError_t (*)(hipGraph_t*,unsigned int) nogil> _hipGraphCreate__funptr)(pGraph,flags)


cdef void* _hipGraphDestroy__funptr = NULL
# 
# @brief Destroys a graph
# 
# @param [in] graph - instance of graph to destroy.
# 
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphDestroy(hipGraph_t graph) nogil:
    global _hipGraphDestroy__funptr
    __init_symbol(&_hipGraphDestroy__funptr,"hipGraphDestroy")
    return (<hipError_t (*)(hipGraph_t) nogil> _hipGraphDestroy__funptr)(graph)


cdef void* _hipGraphAddDependencies__funptr = NULL
# 
# @brief Adds dependency edges to a graph.
# 
# @param [in] graph - instance of the graph to add dependencies.
# @param [in] from - pointer to the graph nodes with dependenties to add from.
# @param [in] to - pointer to the graph nodes to add dependenties to.
# @param [in] numDependencies - the number of dependencies to add.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphAddDependencies(hipGraph_t graph,hipGraphNode_t * from_,hipGraphNode_t * to,unsigned long numDependencies) nogil:
    global _hipGraphAddDependencies__funptr
    __init_symbol(&_hipGraphAddDependencies__funptr,"hipGraphAddDependencies")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t *,hipGraphNode_t *,unsigned long) nogil> _hipGraphAddDependencies__funptr)(graph,from_,to,numDependencies)


cdef void* _hipGraphRemoveDependencies__funptr = NULL
# 
# @brief Removes dependency edges from a graph.
# 
# @param [in] graph - instance of the graph to remove dependencies.
# @param [in] from - Array of nodes that provide the dependencies.
# @param [in] to - Array of dependent nodes.
# @param [in] numDependencies - the number of dependencies to remove.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphRemoveDependencies(hipGraph_t graph,hipGraphNode_t * from_,hipGraphNode_t * to,unsigned long numDependencies) nogil:
    global _hipGraphRemoveDependencies__funptr
    __init_symbol(&_hipGraphRemoveDependencies__funptr,"hipGraphRemoveDependencies")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t *,hipGraphNode_t *,unsigned long) nogil> _hipGraphRemoveDependencies__funptr)(graph,from_,to,numDependencies)


cdef void* _hipGraphGetEdges__funptr = NULL
# 
# @brief Returns a graph's dependency edges.
# 
# @param [in] graph - instance of the graph to get the edges from.
# @param [out] from - pointer to the graph nodes to return edge endpoints.
# @param [out] to - pointer to the graph nodes to return edge endpoints.
# @param [out] numEdges - returns number of edges.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# from and to may both be NULL, in which case this function only returns the number of edges in
# numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher than the actual
# number of edges, the remaining entries in from and to will be set to NULL, and the number of
# edges actually returned will be written to numEdges
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphGetEdges(hipGraph_t graph,hipGraphNode_t* from_,hipGraphNode_t* to,unsigned long * numEdges) nogil:
    global _hipGraphGetEdges__funptr
    __init_symbol(&_hipGraphGetEdges__funptr,"hipGraphGetEdges")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t*,hipGraphNode_t*,unsigned long *) nogil> _hipGraphGetEdges__funptr)(graph,from_,to,numEdges)


cdef void* _hipGraphGetNodes__funptr = NULL
# 
# @brief Returns graph nodes.
# 
# @param [in] graph - instance of graph to get the nodes.
# @param [out] nodes - pointer to return the  graph nodes.
# @param [out] numNodes - returns number of graph nodes.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# nodes may be NULL, in which case this function will return the number of nodes in numNodes.
# Otherwise, numNodes entries will be filled in. If numNodes is higher than the actual number of
# nodes, the remaining entries in nodes will be set to NULL, and the number of nodes actually
# obtained will be returned in numNodes.
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphGetNodes(hipGraph_t graph,hipGraphNode_t* nodes,unsigned long * numNodes) nogil:
    global _hipGraphGetNodes__funptr
    __init_symbol(&_hipGraphGetNodes__funptr,"hipGraphGetNodes")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t*,unsigned long *) nogil> _hipGraphGetNodes__funptr)(graph,nodes,numNodes)


cdef void* _hipGraphGetRootNodes__funptr = NULL
# 
# @brief Returns graph's root nodes.
# 
# @param [in] graph - instance of the graph to get the nodes.
# @param [out] pRootNodes - pointer to return the graph's root nodes.
# @param [out] pNumRootNodes - returns the number of graph's root nodes.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# pRootNodes may be NULL, in which case this function will return the number of root nodes in
# pNumRootNodes. Otherwise, pNumRootNodes entries will be filled in. If pNumRootNodes is higher
# than the actual number of root nodes, the remaining entries in pRootNodes will be set to NULL,
# and the number of nodes actually obtained will be returned in pNumRootNodes.
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphGetRootNodes(hipGraph_t graph,hipGraphNode_t* pRootNodes,unsigned long * pNumRootNodes) nogil:
    global _hipGraphGetRootNodes__funptr
    __init_symbol(&_hipGraphGetRootNodes__funptr,"hipGraphGetRootNodes")
    return (<hipError_t (*)(hipGraph_t,hipGraphNode_t*,unsigned long *) nogil> _hipGraphGetRootNodes__funptr)(graph,pRootNodes,pNumRootNodes)


cdef void* _hipGraphNodeGetDependencies__funptr = NULL
# 
# @brief Returns a node's dependencies.
# 
# @param [in] node - graph node to get the dependencies from.
# @param [out] pDependencies - pointer to to return the dependencies.
# @param [out] pNumDependencies -  returns the number of graph node dependencies.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# pDependencies may be NULL, in which case this function will return the number of dependencies in
# pNumDependencies. Otherwise, pNumDependencies entries will be filled in. If pNumDependencies is
# higher than the actual number of dependencies, the remaining entries in pDependencies will be set
# to NULL, and the number of nodes actually obtained will be returned in pNumDependencies.
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node,hipGraphNode_t* pDependencies,unsigned long * pNumDependencies) nogil:
    global _hipGraphNodeGetDependencies__funptr
    __init_symbol(&_hipGraphNodeGetDependencies__funptr,"hipGraphNodeGetDependencies")
    return (<hipError_t (*)(hipGraphNode_t,hipGraphNode_t*,unsigned long *) nogil> _hipGraphNodeGetDependencies__funptr)(node,pDependencies,pNumDependencies)


cdef void* _hipGraphNodeGetDependentNodes__funptr = NULL
# 
# @brief Returns a node's dependent nodes.
# 
# @param [in] node - graph node to get the Dependent nodes from.
# @param [out] pDependentNodes - pointer to return the graph dependent nodes.
# @param [out] pNumDependentNodes - returns the number of graph node dependent nodes.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# DependentNodes may be NULL, in which case this function will return the number of dependent nodes
# in pNumDependentNodes. Otherwise, pNumDependentNodes entries will be filled in. If
# pNumDependentNodes is higher than the actual number of dependent nodes, the remaining entries in
# pDependentNodes will be set to NULL, and the number of nodes actually obtained will be returned
# in pNumDependentNodes.
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node,hipGraphNode_t* pDependentNodes,unsigned long * pNumDependentNodes) nogil:
    global _hipGraphNodeGetDependentNodes__funptr
    __init_symbol(&_hipGraphNodeGetDependentNodes__funptr,"hipGraphNodeGetDependentNodes")
    return (<hipError_t (*)(hipGraphNode_t,hipGraphNode_t*,unsigned long *) nogil> _hipGraphNodeGetDependentNodes__funptr)(node,pDependentNodes,pNumDependentNodes)


cdef void* _hipGraphNodeGetType__funptr = NULL
# 
# @brief Returns a node's type.
# 
# @param [in] node - instance of the graph to add dependencies.
# @param [out] pType - pointer to the return the type
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphNodeGetType(hipGraphNode_t node,hipGraphNodeType * pType) nogil:
    global _hipGraphNodeGetType__funptr
    __init_symbol(&_hipGraphNodeGetType__funptr,"hipGraphNodeGetType")
    return (<hipError_t (*)(hipGraphNode_t,hipGraphNodeType *) nogil> _hipGraphNodeGetType__funptr)(node,pType)


cdef void* _hipGraphDestroyNode__funptr = NULL
# 
# @brief Remove a node from the graph.
# 
# @param [in] node - graph node to remove
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphDestroyNode(hipGraphNode_t node) nogil:
    global _hipGraphDestroyNode__funptr
    __init_symbol(&_hipGraphDestroyNode__funptr,"hipGraphDestroyNode")
    return (<hipError_t (*)(hipGraphNode_t) nogil> _hipGraphDestroyNode__funptr)(node)


cdef void* _hipGraphClone__funptr = NULL
# 
# @brief Clones a graph.
# 
# @param [out] pGraphClone - Returns newly created cloned graph.
# @param [in] originalGraph - original graph to clone from.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphClone(hipGraph_t* pGraphClone,hipGraph_t originalGraph) nogil:
    global _hipGraphClone__funptr
    __init_symbol(&_hipGraphClone__funptr,"hipGraphClone")
    return (<hipError_t (*)(hipGraph_t*,hipGraph_t) nogil> _hipGraphClone__funptr)(pGraphClone,originalGraph)


cdef void* _hipGraphNodeFindInClone__funptr = NULL
# 
# @brief Finds a cloned version of a node.
# 
# @param [out] pNode - Returns the cloned node.
# @param [in] originalNode - original node handle.
# @param [in] clonedGraph - Cloned graph to query.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode,hipGraphNode_t originalNode,hipGraph_t clonedGraph) nogil:
    global _hipGraphNodeFindInClone__funptr
    __init_symbol(&_hipGraphNodeFindInClone__funptr,"hipGraphNodeFindInClone")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraphNode_t,hipGraph_t) nogil> _hipGraphNodeFindInClone__funptr)(pNode,originalNode,clonedGraph)


cdef void* _hipGraphInstantiate__funptr = NULL
# 
# @brief Creates an executable graph from a graph
# 
# @param [out] pGraphExec - pointer to instantiated executable graph that is created.
# @param [in] graph - instance of graph to instantiate.
# @param [out] pErrorNode - pointer to error node in case error occured in graph instantiation,
#  it could modify the correponding node.
# @param [out] pLogBuffer - pointer to log buffer.
# @param [out] bufferSize - the size of log buffer.
# 
# @returns #hipSuccess, #hipErrorOutOfMemory
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec,hipGraph_t graph,hipGraphNode_t* pErrorNode,char * pLogBuffer,unsigned long bufferSize) nogil:
    global _hipGraphInstantiate__funptr
    __init_symbol(&_hipGraphInstantiate__funptr,"hipGraphInstantiate")
    return (<hipError_t (*)(hipGraphExec_t*,hipGraph_t,hipGraphNode_t*,char *,unsigned long) nogil> _hipGraphInstantiate__funptr)(pGraphExec,graph,pErrorNode,pLogBuffer,bufferSize)


cdef void* _hipGraphInstantiateWithFlags__funptr = NULL
# 
# @brief Creates an executable graph from a graph.
# 
# @param [out] pGraphExec - pointer to instantiated executable graph that is created.
# @param [in] graph - instance of graph to instantiate.
# @param [in] flags - Flags to control instantiation.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
#
cdef hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec,hipGraph_t graph,unsigned long long flags) nogil:
    global _hipGraphInstantiateWithFlags__funptr
    __init_symbol(&_hipGraphInstantiateWithFlags__funptr,"hipGraphInstantiateWithFlags")
    return (<hipError_t (*)(hipGraphExec_t*,hipGraph_t,unsigned long long) nogil> _hipGraphInstantiateWithFlags__funptr)(pGraphExec,graph,flags)


cdef void* _hipGraphLaunch__funptr = NULL
# 
# @brief launches an executable graph in a stream
# 
# @param [in] graphExec - instance of executable graph to launch.
# @param [in] stream - instance of stream in which to launch executable graph.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphLaunch(hipGraphExec_t graphExec,hipStream_t stream) nogil:
    global _hipGraphLaunch__funptr
    __init_symbol(&_hipGraphLaunch__funptr,"hipGraphLaunch")
    return (<hipError_t (*)(hipGraphExec_t,hipStream_t) nogil> _hipGraphLaunch__funptr)(graphExec,stream)


cdef void* _hipGraphUpload__funptr = NULL
# 
# @brief uploads an executable graph in a stream
# 
# @param [in] graphExec - instance of executable graph to launch.
# @param [in] stream - instance of stream in which to launch executable graph.
# @returns #hipSuccess, #hipErrorInvalidValue
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphUpload(hipGraphExec_t graphExec,hipStream_t stream) nogil:
    global _hipGraphUpload__funptr
    __init_symbol(&_hipGraphUpload__funptr,"hipGraphUpload")
    return (<hipError_t (*)(hipGraphExec_t,hipStream_t) nogil> _hipGraphUpload__funptr)(graphExec,stream)


cdef void* _hipGraphExecDestroy__funptr = NULL
# 
# @brief Destroys an executable graph
# 
# @param [in] pGraphExec - instance of executable graph to destry.
# 
# @returns #hipSuccess.
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) nogil:
    global _hipGraphExecDestroy__funptr
    __init_symbol(&_hipGraphExecDestroy__funptr,"hipGraphExecDestroy")
    return (<hipError_t (*)(hipGraphExec_t) nogil> _hipGraphExecDestroy__funptr)(graphExec)


cdef void* _hipGraphExecUpdate__funptr = NULL
# 
# @brief Check whether an executable graph can be updated with a graph and perform the update if  *
# possible.
# 
# @param [in] hGraphExec - instance of executable graph to update.
# @param [in] hGraph - graph that contains the updated parameters.
# @param [in] hErrorNode_out -  node which caused the permissibility check to forbid the update.
# @param [in] updateResult_out - Whether the graph update was permitted.
# @returns #hipSuccess, #hipErrorGraphExecUpdateFailure
# 
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec,hipGraph_t hGraph,hipGraphNode_t* hErrorNode_out,hipGraphExecUpdateResult * updateResult_out) nogil:
    global _hipGraphExecUpdate__funptr
    __init_symbol(&_hipGraphExecUpdate__funptr,"hipGraphExecUpdate")
    return (<hipError_t (*)(hipGraphExec_t,hipGraph_t,hipGraphNode_t*,hipGraphExecUpdateResult *) nogil> _hipGraphExecUpdate__funptr)(hGraphExec,hGraph,hErrorNode_out,updateResult_out)


cdef void* _hipGraphAddKernelNode__funptr = NULL
# 
# @brief Creates a kernel execution node and adds it to a graph.
# 
# @param [out] pGraphNode - pointer to graph node to create.
# @param [in] graph - instance of graph to add the created node.
# @param [in] pDependencies - pointer to the dependencies on the kernel execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] pNodeParams - pointer to the parameters to the kernel execution node on the GPU.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDeviceFunction
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,hipKernelNodeParams * pNodeParams) nogil:
    global _hipGraphAddKernelNode__funptr
    __init_symbol(&_hipGraphAddKernelNode__funptr,"hipGraphAddKernelNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,hipKernelNodeParams *) nogil> _hipGraphAddKernelNode__funptr)(pGraphNode,graph,pDependencies,numDependencies,pNodeParams)


cdef void* _hipGraphKernelNodeGetParams__funptr = NULL
# 
# @brief Gets kernel node's parameters.
# 
# @param [in] node - instance of the node to get parameters from.
# @param [out] pNodeParams - pointer to the parameters
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil:
    global _hipGraphKernelNodeGetParams__funptr
    __init_symbol(&_hipGraphKernelNodeGetParams__funptr,"hipGraphKernelNodeGetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipKernelNodeParams *) nogil> _hipGraphKernelNodeGetParams__funptr)(node,pNodeParams)


cdef void* _hipGraphKernelNodeSetParams__funptr = NULL
# 
# @brief Sets a kernel node's parameters.
# 
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - const pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil:
    global _hipGraphKernelNodeSetParams__funptr
    __init_symbol(&_hipGraphKernelNodeSetParams__funptr,"hipGraphKernelNodeSetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipKernelNodeParams *) nogil> _hipGraphKernelNodeSetParams__funptr)(node,pNodeParams)


cdef void* _hipGraphExecKernelNodeSetParams__funptr = NULL
# 
# @brief Sets the parameters for a kernel node in the given graphExec.
# 
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - const pointer to the kernel node parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil:
    global _hipGraphExecKernelNodeSetParams__funptr
    __init_symbol(&_hipGraphExecKernelNodeSetParams__funptr,"hipGraphExecKernelNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipKernelNodeParams *) nogil> _hipGraphExecKernelNodeSetParams__funptr)(hGraphExec,node,pNodeParams)


cdef void* _hipGraphAddMemcpyNode__funptr = NULL
# 
# @brief Creates a memcpy node and adds it to a graph.
# 
# @param [out] pGraphNode - pointer to graph node to create.
# @param [in] graph - instance of graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] pCopyParams - const pointer to the parameters for the memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,hipMemcpy3DParms * pCopyParams) nogil:
    global _hipGraphAddMemcpyNode__funptr
    __init_symbol(&_hipGraphAddMemcpyNode__funptr,"hipGraphAddMemcpyNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,hipMemcpy3DParms *) nogil> _hipGraphAddMemcpyNode__funptr)(pGraphNode,graph,pDependencies,numDependencies,pCopyParams)


cdef void* _hipGraphMemcpyNodeGetParams__funptr = NULL
# 
# @brief Gets a memcpy node's parameters.
# 
# @param [in] node - instance of the node to get parameters from.
# @param [out] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil:
    global _hipGraphMemcpyNodeGetParams__funptr
    __init_symbol(&_hipGraphMemcpyNodeGetParams__funptr,"hipGraphMemcpyNodeGetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipMemcpy3DParms *) nogil> _hipGraphMemcpyNodeGetParams__funptr)(node,pNodeParams)


cdef void* _hipGraphMemcpyNodeSetParams__funptr = NULL
# 
# @brief Sets a memcpy node's parameters.
# 
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - const pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil:
    global _hipGraphMemcpyNodeSetParams__funptr
    __init_symbol(&_hipGraphMemcpyNodeSetParams__funptr,"hipGraphMemcpyNodeSetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipMemcpy3DParms *) nogil> _hipGraphMemcpyNodeSetParams__funptr)(node,pNodeParams)


cdef void* _hipGraphKernelNodeSetAttribute__funptr = NULL
# 
# @brief Sets a node attribute.
# 
# @param [in] hNode - instance of the node to set parameters to.
# @param [in] attr - the attribute node is set to.
# @param [in] value - const pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode,hipKernelNodeAttrID attr,hipKernelNodeAttrValue * value) nogil:
    global _hipGraphKernelNodeSetAttribute__funptr
    __init_symbol(&_hipGraphKernelNodeSetAttribute__funptr,"hipGraphKernelNodeSetAttribute")
    return (<hipError_t (*)(hipGraphNode_t,hipKernelNodeAttrID,hipKernelNodeAttrValue *) nogil> _hipGraphKernelNodeSetAttribute__funptr)(hNode,attr,value)


cdef void* _hipGraphKernelNodeGetAttribute__funptr = NULL
# 
# @brief Gets a node attribute.
# 
# @param [in] hNode - instance of the node to set parameters to.
# @param [in] attr - the attribute node is set to.
# @param [in] value - const pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode,hipKernelNodeAttrID attr,hipKernelNodeAttrValue * value) nogil:
    global _hipGraphKernelNodeGetAttribute__funptr
    __init_symbol(&_hipGraphKernelNodeGetAttribute__funptr,"hipGraphKernelNodeGetAttribute")
    return (<hipError_t (*)(hipGraphNode_t,hipKernelNodeAttrID,hipKernelNodeAttrValue *) nogil> _hipGraphKernelNodeGetAttribute__funptr)(hNode,attr,value)


cdef void* _hipGraphExecMemcpyNodeSetParams__funptr = NULL
# 
# @brief Sets the parameters for a memcpy node in the given graphExec.
# 
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - const pointer to the kernel node parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil:
    global _hipGraphExecMemcpyNodeSetParams__funptr
    __init_symbol(&_hipGraphExecMemcpyNodeSetParams__funptr,"hipGraphExecMemcpyNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipMemcpy3DParms *) nogil> _hipGraphExecMemcpyNodeSetParams__funptr)(hGraphExec,node,pNodeParams)


cdef void* _hipGraphAddMemcpyNode1D__funptr = NULL
# 
# @brief Creates a 1D memcpy node and adds it to a graph.
# 
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
cdef hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,void * dst,const void * src,unsigned long count,hipMemcpyKind kind) nogil:
    global _hipGraphAddMemcpyNode1D__funptr
    __init_symbol(&_hipGraphAddMemcpyNode1D__funptr,"hipGraphAddMemcpyNode1D")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,void *,const void *,unsigned long,hipMemcpyKind) nogil> _hipGraphAddMemcpyNode1D__funptr)(pGraphNode,graph,pDependencies,numDependencies,dst,src,count,kind)


cdef void* _hipGraphMemcpyNodeSetParams1D__funptr = NULL
# 
# @brief Sets a memcpy node's parameters to perform a 1-dimensional copy.
# 
# @param [in] node - instance of the node to set parameters to.
# @param [in] dst - pointer to memory address to the destination.
# @param [in] src - pointer to memory address to the source.
# @param [in] count - the size of the memory to copy.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node,void * dst,const void * src,unsigned long count,hipMemcpyKind kind) nogil:
    global _hipGraphMemcpyNodeSetParams1D__funptr
    __init_symbol(&_hipGraphMemcpyNodeSetParams1D__funptr,"hipGraphMemcpyNodeSetParams1D")
    return (<hipError_t (*)(hipGraphNode_t,void *,const void *,unsigned long,hipMemcpyKind) nogil> _hipGraphMemcpyNodeSetParams1D__funptr)(node,dst,src,count,kind)


cdef void* _hipGraphExecMemcpyNodeSetParams1D__funptr = NULL
# 
# @brief Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional
# copy.
# 
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] dst - pointer to memory address to the destination.
# @param [in] src - pointer to memory address to the source.
# @param [in] count - the size of the memory to copy.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec,hipGraphNode_t node,void * dst,const void * src,unsigned long count,hipMemcpyKind kind) nogil:
    global _hipGraphExecMemcpyNodeSetParams1D__funptr
    __init_symbol(&_hipGraphExecMemcpyNodeSetParams1D__funptr,"hipGraphExecMemcpyNodeSetParams1D")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,unsigned long,hipMemcpyKind) nogil> _hipGraphExecMemcpyNodeSetParams1D__funptr)(hGraphExec,node,dst,src,count,kind)


cdef void* _hipGraphAddMemcpyNodeFromSymbol__funptr = NULL
# 
# @brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph.
# 
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
cdef hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,void * dst,const void * symbol,unsigned long count,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipGraphAddMemcpyNodeFromSymbol__funptr
    __init_symbol(&_hipGraphAddMemcpyNodeFromSymbol__funptr,"hipGraphAddMemcpyNodeFromSymbol")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipGraphAddMemcpyNodeFromSymbol__funptr)(pGraphNode,graph,pDependencies,numDependencies,dst,symbol,count,offset,kind)


cdef void* _hipGraphMemcpyNodeSetParamsFromSymbol__funptr = NULL
# 
# @brief Sets a memcpy node's parameters to copy from a symbol on the device.
# 
# @param [in] node - instance of the node to set parameters to.
# @param [in] dst - pointer to memory address to the destination.
# @param [in] symbol - Device symbol address.
# @param [in] count - the size of the memory to copy.
# @param [in] offset - Offset from start of symbol in bytes.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node,void * dst,const void * symbol,unsigned long count,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipGraphMemcpyNodeSetParamsFromSymbol__funptr
    __init_symbol(&_hipGraphMemcpyNodeSetParamsFromSymbol__funptr,"hipGraphMemcpyNodeSetParamsFromSymbol")
    return (<hipError_t (*)(hipGraphNode_t,void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipGraphMemcpyNodeSetParamsFromSymbol__funptr)(node,dst,symbol,count,offset,kind)


cdef void* _hipGraphExecMemcpyNodeSetParamsFromSymbol__funptr = NULL
# 
# @brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the
# * device.
# 
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
cdef hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec,hipGraphNode_t node,void * dst,const void * symbol,unsigned long count,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipGraphExecMemcpyNodeSetParamsFromSymbol__funptr
    __init_symbol(&_hipGraphExecMemcpyNodeSetParamsFromSymbol__funptr,"hipGraphExecMemcpyNodeSetParamsFromSymbol")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipGraphExecMemcpyNodeSetParamsFromSymbol__funptr)(hGraphExec,node,dst,symbol,count,offset,kind)


cdef void* _hipGraphAddMemcpyNodeToSymbol__funptr = NULL
# 
# @brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph.
# 
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
cdef hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,const void * symbol,const void * src,unsigned long count,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipGraphAddMemcpyNodeToSymbol__funptr
    __init_symbol(&_hipGraphAddMemcpyNodeToSymbol__funptr,"hipGraphAddMemcpyNodeToSymbol")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,const void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipGraphAddMemcpyNodeToSymbol__funptr)(pGraphNode,graph,pDependencies,numDependencies,symbol,src,count,offset,kind)


cdef void* _hipGraphMemcpyNodeSetParamsToSymbol__funptr = NULL
# 
# @brief Sets a memcpy node's parameters to copy to a symbol on the device.
# 
# @param [in] node - instance of the node to set parameters to.
# @param [in] symbol - Device symbol address.
# @param [in] src - pointer to memory address of the src.
# @param [in] count - the size of the memory to copy.
# @param [in] offset - Offset from start of symbol in bytes.
# @param [in] kind - the type of memory copy.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node,const void * symbol,const void * src,unsigned long count,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipGraphMemcpyNodeSetParamsToSymbol__funptr
    __init_symbol(&_hipGraphMemcpyNodeSetParamsToSymbol__funptr,"hipGraphMemcpyNodeSetParamsToSymbol")
    return (<hipError_t (*)(hipGraphNode_t,const void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipGraphMemcpyNodeSetParamsToSymbol__funptr)(node,symbol,src,count,offset,kind)


cdef void* _hipGraphExecMemcpyNodeSetParamsToSymbol__funptr = NULL
# 
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
cdef hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec,hipGraphNode_t node,const void * symbol,const void * src,unsigned long count,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipGraphExecMemcpyNodeSetParamsToSymbol__funptr
    __init_symbol(&_hipGraphExecMemcpyNodeSetParamsToSymbol__funptr,"hipGraphExecMemcpyNodeSetParamsToSymbol")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,const void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipGraphExecMemcpyNodeSetParamsToSymbol__funptr)(hGraphExec,node,symbol,src,count,offset,kind)


cdef void* _hipGraphAddMemsetNode__funptr = NULL
# 
# @brief Creates a memset node and adds it to a graph.
# 
# @param [out] pGraphNode - pointer to the graph node to create.
# @param [in] graph - instance of the graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] pMemsetParams - const pointer to the parameters for the memory set.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,hipMemsetParams * pMemsetParams) nogil:
    global _hipGraphAddMemsetNode__funptr
    __init_symbol(&_hipGraphAddMemsetNode__funptr,"hipGraphAddMemsetNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,hipMemsetParams *) nogil> _hipGraphAddMemsetNode__funptr)(pGraphNode,graph,pDependencies,numDependencies,pMemsetParams)


cdef void* _hipGraphMemsetNodeGetParams__funptr = NULL
# 
# @brief Gets a memset node's parameters.
# 
# @param [in] node - instane of the node to get parameters from.
# @param [out] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil:
    global _hipGraphMemsetNodeGetParams__funptr
    __init_symbol(&_hipGraphMemsetNodeGetParams__funptr,"hipGraphMemsetNodeGetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipMemsetParams *) nogil> _hipGraphMemsetNodeGetParams__funptr)(node,pNodeParams)


cdef void* _hipGraphMemsetNodeSetParams__funptr = NULL
# 
# @brief Sets a memset node's parameters.
# 
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil:
    global _hipGraphMemsetNodeSetParams__funptr
    __init_symbol(&_hipGraphMemsetNodeSetParams__funptr,"hipGraphMemsetNodeSetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipMemsetParams *) nogil> _hipGraphMemsetNodeSetParams__funptr)(node,pNodeParams)


cdef void* _hipGraphExecMemsetNodeSetParams__funptr = NULL
# 
# @brief Sets the parameters for a memset node in the given graphExec.
# 
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil:
    global _hipGraphExecMemsetNodeSetParams__funptr
    __init_symbol(&_hipGraphExecMemsetNodeSetParams__funptr,"hipGraphExecMemsetNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipMemsetParams *) nogil> _hipGraphExecMemsetNodeSetParams__funptr)(hGraphExec,node,pNodeParams)


cdef void* _hipGraphAddHostNode__funptr = NULL
# 
# @brief Creates a host execution node and adds it to a graph.
# 
# @param [out] pGraphNode - pointer to the graph node to create.
# @param [in] graph - instance of the graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] pNodeParams -pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,hipHostNodeParams * pNodeParams) nogil:
    global _hipGraphAddHostNode__funptr
    __init_symbol(&_hipGraphAddHostNode__funptr,"hipGraphAddHostNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,hipHostNodeParams *) nogil> _hipGraphAddHostNode__funptr)(pGraphNode,graph,pDependencies,numDependencies,pNodeParams)


cdef void* _hipGraphHostNodeGetParams__funptr = NULL
# 
# @brief Returns a host node's parameters.
# 
# @param [in] node - instane of the node to get parameters from.
# @param [out] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil:
    global _hipGraphHostNodeGetParams__funptr
    __init_symbol(&_hipGraphHostNodeGetParams__funptr,"hipGraphHostNodeGetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipHostNodeParams *) nogil> _hipGraphHostNodeGetParams__funptr)(node,pNodeParams)


cdef void* _hipGraphHostNodeSetParams__funptr = NULL
# 
# @brief Sets a host node's parameters.
# 
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil:
    global _hipGraphHostNodeSetParams__funptr
    __init_symbol(&_hipGraphHostNodeSetParams__funptr,"hipGraphHostNodeSetParams")
    return (<hipError_t (*)(hipGraphNode_t,hipHostNodeParams *) nogil> _hipGraphHostNodeSetParams__funptr)(node,pNodeParams)


cdef void* _hipGraphExecHostNodeSetParams__funptr = NULL
# 
# @brief Sets the parameters for a host node in the given graphExec.
# 
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - instance of the node to set parameters to.
# @param [in] pNodeParams - pointer to the parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil:
    global _hipGraphExecHostNodeSetParams__funptr
    __init_symbol(&_hipGraphExecHostNodeSetParams__funptr,"hipGraphExecHostNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipHostNodeParams *) nogil> _hipGraphExecHostNodeSetParams__funptr)(hGraphExec,node,pNodeParams)


cdef void* _hipGraphAddChildGraphNode__funptr = NULL
# 
# @brief Creates a child graph node and adds it to a graph.
# 
# @param [out] pGraphNode - pointer to the graph node to create.
# @param [in] graph - instance of the graph to add the created node.
# @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
# @param [in] numDependencies - the number of the dependencies.
# @param [in] childGraph - the graph to clone into this node
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,hipGraph_t childGraph) nogil:
    global _hipGraphAddChildGraphNode__funptr
    __init_symbol(&_hipGraphAddChildGraphNode__funptr,"hipGraphAddChildGraphNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,hipGraph_t) nogil> _hipGraphAddChildGraphNode__funptr)(pGraphNode,graph,pDependencies,numDependencies,childGraph)


cdef void* _hipGraphChildGraphNodeGetGraph__funptr = NULL
# 
# @brief Gets a handle to the embedded graph of a child graph node.
# 
# @param [in] node - instane of the node to get child graph.
# @param [out] pGraph - pointer to get the graph.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node,hipGraph_t* pGraph) nogil:
    global _hipGraphChildGraphNodeGetGraph__funptr
    __init_symbol(&_hipGraphChildGraphNodeGetGraph__funptr,"hipGraphChildGraphNodeGetGraph")
    return (<hipError_t (*)(hipGraphNode_t,hipGraph_t*) nogil> _hipGraphChildGraphNodeGetGraph__funptr)(node,pGraph)


cdef void* _hipGraphExecChildGraphNodeSetParams__funptr = NULL
# 
# @brief Updates node parameters in the child graph node in the given graphExec.
# 
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] node - node from the graph which was used to instantiate graphExec.
# @param [in] childGraph - child graph with updated parameters.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipGraph_t childGraph) nogil:
    global _hipGraphExecChildGraphNodeSetParams__funptr
    __init_symbol(&_hipGraphExecChildGraphNodeSetParams__funptr,"hipGraphExecChildGraphNodeSetParams")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipGraph_t) nogil> _hipGraphExecChildGraphNodeSetParams__funptr)(hGraphExec,node,childGraph)


cdef void* _hipGraphAddEmptyNode__funptr = NULL
# 
# @brief Creates an empty node and adds it to a graph.
# 
# @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
# @param [in] graph - instane of the graph the node is add to.
# @param [in] pDependencies - const pointer to the node dependenties.
# @param [in] numDependencies - the number of dependencies.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies) nogil:
    global _hipGraphAddEmptyNode__funptr
    __init_symbol(&_hipGraphAddEmptyNode__funptr,"hipGraphAddEmptyNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long) nogil> _hipGraphAddEmptyNode__funptr)(pGraphNode,graph,pDependencies,numDependencies)


cdef void* _hipGraphAddEventRecordNode__funptr = NULL
# 
# @brief Creates an event record node and adds it to a graph.
# 
# @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
# @param [in] graph - instane of the graph the node to be added.
# @param [in] pDependencies - const pointer to the node dependenties.
# @param [in] numDependencies - the number of dependencies.
# @param [in] event - Event for the node.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,hipEvent_t event) nogil:
    global _hipGraphAddEventRecordNode__funptr
    __init_symbol(&_hipGraphAddEventRecordNode__funptr,"hipGraphAddEventRecordNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,hipEvent_t) nogil> _hipGraphAddEventRecordNode__funptr)(pGraphNode,graph,pDependencies,numDependencies,event)


cdef void* _hipGraphEventRecordNodeGetEvent__funptr = NULL
# 
# @brief Returns the event associated with an event record node.
# 
# @param [in] node -  instane of the node to get event from.
# @param [out] event_out - Pointer to return the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node,hipEvent_t* event_out) nogil:
    global _hipGraphEventRecordNodeGetEvent__funptr
    __init_symbol(&_hipGraphEventRecordNodeGetEvent__funptr,"hipGraphEventRecordNodeGetEvent")
    return (<hipError_t (*)(hipGraphNode_t,hipEvent_t*) nogil> _hipGraphEventRecordNodeGetEvent__funptr)(node,event_out)


cdef void* _hipGraphEventRecordNodeSetEvent__funptr = NULL
# 
# @brief Sets an event record node's event.
# 
# @param [in] node - instane of the node to set event to.
# @param [in] event - pointer to the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node,hipEvent_t event) nogil:
    global _hipGraphEventRecordNodeSetEvent__funptr
    __init_symbol(&_hipGraphEventRecordNodeSetEvent__funptr,"hipGraphEventRecordNodeSetEvent")
    return (<hipError_t (*)(hipGraphNode_t,hipEvent_t) nogil> _hipGraphEventRecordNodeSetEvent__funptr)(node,event)


cdef void* _hipGraphExecEventRecordNodeSetEvent__funptr = NULL
# 
# @brief Sets the event for an event record node in the given graphExec.
# 
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] hNode - node from the graph which was used to instantiate graphExec.
# @param [in] event - pointer to the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec,hipGraphNode_t hNode,hipEvent_t event) nogil:
    global _hipGraphExecEventRecordNodeSetEvent__funptr
    __init_symbol(&_hipGraphExecEventRecordNodeSetEvent__funptr,"hipGraphExecEventRecordNodeSetEvent")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t) nogil> _hipGraphExecEventRecordNodeSetEvent__funptr)(hGraphExec,hNode,event)


cdef void* _hipGraphAddEventWaitNode__funptr = NULL
# 
# @brief Creates an event wait node and adds it to a graph.
# 
# @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
# @param [in] graph - instane of the graph the node to be added.
# @param [in] pDependencies - const pointer to the node dependenties.
# @param [in] numDependencies - the number of dependencies.
# @param [in] event - Event for the node.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,unsigned long numDependencies,hipEvent_t event) nogil:
    global _hipGraphAddEventWaitNode__funptr
    __init_symbol(&_hipGraphAddEventWaitNode__funptr,"hipGraphAddEventWaitNode")
    return (<hipError_t (*)(hipGraphNode_t*,hipGraph_t,hipGraphNode_t *,unsigned long,hipEvent_t) nogil> _hipGraphAddEventWaitNode__funptr)(pGraphNode,graph,pDependencies,numDependencies,event)


cdef void* _hipGraphEventWaitNodeGetEvent__funptr = NULL
# 
# @brief Returns the event associated with an event wait node.
# 
# @param [in] node -  instane of the node to get event from.
# @param [out] event_out - Pointer to return the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node,hipEvent_t* event_out) nogil:
    global _hipGraphEventWaitNodeGetEvent__funptr
    __init_symbol(&_hipGraphEventWaitNodeGetEvent__funptr,"hipGraphEventWaitNodeGetEvent")
    return (<hipError_t (*)(hipGraphNode_t,hipEvent_t*) nogil> _hipGraphEventWaitNodeGetEvent__funptr)(node,event_out)


cdef void* _hipGraphEventWaitNodeSetEvent__funptr = NULL
# 
# @brief Sets an event wait node's event.
# 
# @param [in] node - instane of the node to set event to.
# @param [in] event - pointer to the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node,hipEvent_t event) nogil:
    global _hipGraphEventWaitNodeSetEvent__funptr
    __init_symbol(&_hipGraphEventWaitNodeSetEvent__funptr,"hipGraphEventWaitNodeSetEvent")
    return (<hipError_t (*)(hipGraphNode_t,hipEvent_t) nogil> _hipGraphEventWaitNodeSetEvent__funptr)(node,event)


cdef void* _hipGraphExecEventWaitNodeSetEvent__funptr = NULL
# 
# @brief Sets the event for an event record node in the given graphExec.
# 
# @param [in] hGraphExec - instance of the executable graph with the node.
# @param [in] hNode - node from the graph which was used to instantiate graphExec.
# @param [in] event - pointer to the event.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec,hipGraphNode_t hNode,hipEvent_t event) nogil:
    global _hipGraphExecEventWaitNodeSetEvent__funptr
    __init_symbol(&_hipGraphExecEventWaitNodeSetEvent__funptr,"hipGraphExecEventWaitNodeSetEvent")
    return (<hipError_t (*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t) nogil> _hipGraphExecEventWaitNodeSetEvent__funptr)(hGraphExec,hNode,event)


cdef void* _hipDeviceGetGraphMemAttribute__funptr = NULL
# 
# @brief Get the mem attribute for graphs.
# 
# @param [in] device - device the attr is get for.
# @param [in] attr - attr to get.
# @param [out] value - value for specific attr.
# @returns #hipSuccess, #hipErrorInvalidDevice
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceGetGraphMemAttribute(int device,hipGraphMemAttributeType attr,void * value) nogil:
    global _hipDeviceGetGraphMemAttribute__funptr
    __init_symbol(&_hipDeviceGetGraphMemAttribute__funptr,"hipDeviceGetGraphMemAttribute")
    return (<hipError_t (*)(int,hipGraphMemAttributeType,void *) nogil> _hipDeviceGetGraphMemAttribute__funptr)(device,attr,value)


cdef void* _hipDeviceSetGraphMemAttribute__funptr = NULL
# 
# @brief Set the mem attribute for graphs.
# 
# @param [in] device - device the attr is set for.
# @param [in] attr - attr to set.
# @param [in] value - value for specific attr.
# @returns #hipSuccess, #hipErrorInvalidDevice
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceSetGraphMemAttribute(int device,hipGraphMemAttributeType attr,void * value) nogil:
    global _hipDeviceSetGraphMemAttribute__funptr
    __init_symbol(&_hipDeviceSetGraphMemAttribute__funptr,"hipDeviceSetGraphMemAttribute")
    return (<hipError_t (*)(int,hipGraphMemAttributeType,void *) nogil> _hipDeviceSetGraphMemAttribute__funptr)(device,attr,value)


cdef void* _hipDeviceGraphMemTrim__funptr = NULL
# 
# @brief Free unused memory on specific device used for graph back to OS.
# 
# @param [in] device - device the memory is used for graphs
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipDeviceGraphMemTrim(int device) nogil:
    global _hipDeviceGraphMemTrim__funptr
    __init_symbol(&_hipDeviceGraphMemTrim__funptr,"hipDeviceGraphMemTrim")
    return (<hipError_t (*)(int) nogil> _hipDeviceGraphMemTrim__funptr)(device)


cdef void* _hipUserObjectCreate__funptr = NULL
# 
# @brief Create an instance of userObject to manage lifetime of a resource.
# 
# @param [out] object_out - pointer to instace of userobj.
# @param [in] ptr - pointer to pass to destroy function.
# @param [in] destroy - destroy callback to remove resource.
# @param [in] initialRefcount - reference to resource.
# @param [in] flags - flags passed to API.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipUserObjectCreate(hipUserObject_t* object_out,void * ptr,hipHostFn_t destroy,unsigned int initialRefcount,unsigned int flags):
    global _hipUserObjectCreate__funptr
    __init_symbol(&_hipUserObjectCreate__funptr,"hipUserObjectCreate")
    return (<hipError_t (*)(hipUserObject_t*,void *,hipHostFn_t,unsigned int,unsigned int)> _hipUserObjectCreate__funptr)(object_out,ptr,destroy,initialRefcount,flags)


cdef void* _hipUserObjectRelease__funptr = NULL
# 
# @brief Release number of references to resource.
# 
# @param [in] object - pointer to instace of userobj.
# @param [in] count - reference to resource to be retained.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipUserObjectRelease(hipUserObject_t object,unsigned int count) nogil:
    global _hipUserObjectRelease__funptr
    __init_symbol(&_hipUserObjectRelease__funptr,"hipUserObjectRelease")
    return (<hipError_t (*)(hipUserObject_t,unsigned int) nogil> _hipUserObjectRelease__funptr)(object,count)


cdef void* _hipUserObjectRetain__funptr = NULL
# 
# @brief Retain number of references to resource.
# 
# @param [in] object - pointer to instace of userobj.
# @param [in] count - reference to resource to be retained.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipUserObjectRetain(hipUserObject_t object,unsigned int count) nogil:
    global _hipUserObjectRetain__funptr
    __init_symbol(&_hipUserObjectRetain__funptr,"hipUserObjectRetain")
    return (<hipError_t (*)(hipUserObject_t,unsigned int) nogil> _hipUserObjectRetain__funptr)(object,count)


cdef void* _hipGraphRetainUserObject__funptr = NULL
# 
# @brief Retain user object for graphs.
# 
# @param [in] graph - pointer to graph to retain the user object for.
# @param [in] object - pointer to instace of userobj.
# @param [in] count - reference to resource to be retained.
# @param [in] flags - flags passed to API.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphRetainUserObject(hipGraph_t graph,hipUserObject_t object,unsigned int count,unsigned int flags) nogil:
    global _hipGraphRetainUserObject__funptr
    __init_symbol(&_hipGraphRetainUserObject__funptr,"hipGraphRetainUserObject")
    return (<hipError_t (*)(hipGraph_t,hipUserObject_t,unsigned int,unsigned int) nogil> _hipGraphRetainUserObject__funptr)(graph,object,count,flags)


cdef void* _hipGraphReleaseUserObject__funptr = NULL
# 
# @brief Release user object from graphs.
# 
# @param [in] graph - pointer to graph to retain the user object for.
# @param [in] object - pointer to instace of userobj.
# @param [in] count - reference to resource to be retained.
# @returns #hipSuccess, #hipErrorInvalidValue
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipGraphReleaseUserObject(hipGraph_t graph,hipUserObject_t object,unsigned int count) nogil:
    global _hipGraphReleaseUserObject__funptr
    __init_symbol(&_hipGraphReleaseUserObject__funptr,"hipGraphReleaseUserObject")
    return (<hipError_t (*)(hipGraph_t,hipUserObject_t,unsigned int) nogil> _hipGraphReleaseUserObject__funptr)(graph,object,count)


cdef void* _hipMemAddressFree__funptr = NULL
# 
# @brief Frees an address range reservation made via hipMemAddressReserve
# 
# @param [in] devPtr - starting address of the range.
# @param [in] size - size of the range.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemAddressFree(void * devPtr,unsigned long size) nogil:
    global _hipMemAddressFree__funptr
    __init_symbol(&_hipMemAddressFree__funptr,"hipMemAddressFree")
    return (<hipError_t (*)(void *,unsigned long) nogil> _hipMemAddressFree__funptr)(devPtr,size)


cdef void* _hipMemAddressReserve__funptr = NULL
# 
# @brief Reserves an address range
# 
# @param [out] ptr - starting address of the reserved range.
# @param [in] size - size of the reservation.
# @param [in] alignment - alignment of the address.
# @param [in] addr - requested starting address of the range.
# @param [in] flags - currently unused, must be zero.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemAddressReserve(void ** ptr,unsigned long size,unsigned long alignment,void * addr,unsigned long long flags) nogil:
    global _hipMemAddressReserve__funptr
    __init_symbol(&_hipMemAddressReserve__funptr,"hipMemAddressReserve")
    return (<hipError_t (*)(void **,unsigned long,unsigned long,void *,unsigned long long) nogil> _hipMemAddressReserve__funptr)(ptr,size,alignment,addr,flags)


cdef void* _hipMemCreate__funptr = NULL
# 
# @brief Creates a memory allocation described by the properties and size
# 
# @param [out] handle - value of the returned handle.
# @param [in] size - size of the allocation.
# @param [in] prop - properties of the allocation.
# @param [in] flags - currently unused, must be zero.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle,unsigned long size,hipMemAllocationProp * prop,unsigned long long flags) nogil:
    global _hipMemCreate__funptr
    __init_symbol(&_hipMemCreate__funptr,"hipMemCreate")
    return (<hipError_t (*)(hipMemGenericAllocationHandle_t*,unsigned long,hipMemAllocationProp *,unsigned long long) nogil> _hipMemCreate__funptr)(handle,size,prop,flags)


cdef void* _hipMemExportToShareableHandle__funptr = NULL
# 
# @brief Exports an allocation to a requested shareable handle type.
# 
# @param [out] shareableHandle - value of the returned handle.
# @param [in] handle - handle to share.
# @param [in] handleType - type of the shareable handle.
# @param [in] flags - currently unused, must be zero.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemExportToShareableHandle(void * shareableHandle,hipMemGenericAllocationHandle_t handle,hipMemAllocationHandleType handleType,unsigned long long flags) nogil:
    global _hipMemExportToShareableHandle__funptr
    __init_symbol(&_hipMemExportToShareableHandle__funptr,"hipMemExportToShareableHandle")
    return (<hipError_t (*)(void *,hipMemGenericAllocationHandle_t,hipMemAllocationHandleType,unsigned long long) nogil> _hipMemExportToShareableHandle__funptr)(shareableHandle,handle,handleType,flags)


cdef void* _hipMemGetAccess__funptr = NULL
# 
# @brief Get the access flags set for the given location and ptr.
# 
# @param [out] flags - flags for this location.
# @param [in] location - target location.
# @param [in] ptr - address to check the access flags.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemGetAccess(unsigned long long * flags,hipMemLocation * location,void * ptr) nogil:
    global _hipMemGetAccess__funptr
    __init_symbol(&_hipMemGetAccess__funptr,"hipMemGetAccess")
    return (<hipError_t (*)(unsigned long long *,hipMemLocation *,void *) nogil> _hipMemGetAccess__funptr)(flags,location,ptr)


cdef void* _hipMemGetAllocationGranularity__funptr = NULL
# 
# @brief Calculates either the minimal or recommended granularity.
# 
# @param [out] granularity - returned granularity.
# @param [in] prop - location properties.
# @param [in] option - determines which granularity to return.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemGetAllocationGranularity(unsigned long * granularity,hipMemAllocationProp * prop,hipMemAllocationGranularity_flags option) nogil:
    global _hipMemGetAllocationGranularity__funptr
    __init_symbol(&_hipMemGetAllocationGranularity__funptr,"hipMemGetAllocationGranularity")
    return (<hipError_t (*)(unsigned long *,hipMemAllocationProp *,hipMemAllocationGranularity_flags) nogil> _hipMemGetAllocationGranularity__funptr)(granularity,prop,option)


cdef void* _hipMemGetAllocationPropertiesFromHandle__funptr = NULL
# 
# @brief Retrieve the property structure of the given handle.
# 
# @param [out] prop - properties of the given handle.
# @param [in] handle - handle to perform the query on.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp * prop,hipMemGenericAllocationHandle_t handle) nogil:
    global _hipMemGetAllocationPropertiesFromHandle__funptr
    __init_symbol(&_hipMemGetAllocationPropertiesFromHandle__funptr,"hipMemGetAllocationPropertiesFromHandle")
    return (<hipError_t (*)(hipMemAllocationProp *,hipMemGenericAllocationHandle_t) nogil> _hipMemGetAllocationPropertiesFromHandle__funptr)(prop,handle)


cdef void* _hipMemImportFromShareableHandle__funptr = NULL
# 
# @brief Imports an allocation from a requested shareable handle type.
# 
# @param [out] handle - returned value.
# @param [in] osHandle - shareable handle representing the memory allocation.
# @param [in] shHandleType - handle type.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle,void * osHandle,hipMemAllocationHandleType shHandleType) nogil:
    global _hipMemImportFromShareableHandle__funptr
    __init_symbol(&_hipMemImportFromShareableHandle__funptr,"hipMemImportFromShareableHandle")
    return (<hipError_t (*)(hipMemGenericAllocationHandle_t*,void *,hipMemAllocationHandleType) nogil> _hipMemImportFromShareableHandle__funptr)(handle,osHandle,shHandleType)


cdef void* _hipMemMap__funptr = NULL
# 
# @brief Maps an allocation handle to a reserved virtual address range.
# 
# @param [in] ptr - address where the memory will be mapped.
# @param [in] size - size of the mapping.
# @param [in] offset - offset into the memory, currently must be zero.
# @param [in] handle - memory allocation to be mapped.
# @param [in] flags - currently unused, must be zero.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemMap(void * ptr,unsigned long size,unsigned long offset,hipMemGenericAllocationHandle_t handle,unsigned long long flags) nogil:
    global _hipMemMap__funptr
    __init_symbol(&_hipMemMap__funptr,"hipMemMap")
    return (<hipError_t (*)(void *,unsigned long,unsigned long,hipMemGenericAllocationHandle_t,unsigned long long) nogil> _hipMemMap__funptr)(ptr,size,offset,handle,flags)


cdef void* _hipMemMapArrayAsync__funptr = NULL
# 
# @brief Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.
# 
# @param [in] mapInfoList - list of hipArrayMapInfo.
# @param [in] count - number of hipArrayMapInfo in mapInfoList.
# @param [in] stream - stream identifier for the stream to use for map or unmap operations.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemMapArrayAsync(hipArrayMapInfo * mapInfoList,unsigned int count,hipStream_t stream) nogil:
    global _hipMemMapArrayAsync__funptr
    __init_symbol(&_hipMemMapArrayAsync__funptr,"hipMemMapArrayAsync")
    return (<hipError_t (*)(hipArrayMapInfo *,unsigned int,hipStream_t) nogil> _hipMemMapArrayAsync__funptr)(mapInfoList,count,stream)


cdef void* _hipMemRelease__funptr = NULL
# 
# @brief Release a memory handle representing a memory allocation which was previously allocated through hipMemCreate.
# 
# @param [in] handle - handle of the memory allocation.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) nogil:
    global _hipMemRelease__funptr
    __init_symbol(&_hipMemRelease__funptr,"hipMemRelease")
    return (<hipError_t (*)(hipMemGenericAllocationHandle_t) nogil> _hipMemRelease__funptr)(handle)


cdef void* _hipMemRetainAllocationHandle__funptr = NULL
# 
# @brief Returns the allocation handle of the backing memory allocation given the address.
# 
# @param [out] handle - handle representing addr.
# @param [in] addr - address to look up.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle,void * addr) nogil:
    global _hipMemRetainAllocationHandle__funptr
    __init_symbol(&_hipMemRetainAllocationHandle__funptr,"hipMemRetainAllocationHandle")
    return (<hipError_t (*)(hipMemGenericAllocationHandle_t*,void *) nogil> _hipMemRetainAllocationHandle__funptr)(handle,addr)


cdef void* _hipMemSetAccess__funptr = NULL
# 
# @brief Set the access flags for each location specified in desc for the given virtual address range.
# 
# @param [in] ptr - starting address of the virtual address range.
# @param [in] size - size of the range.
# @param [in] desc - array of hipMemAccessDesc.
# @param [in] count - number of hipMemAccessDesc in desc.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemSetAccess(void * ptr,unsigned long size,hipMemAccessDesc * desc,unsigned long count) nogil:
    global _hipMemSetAccess__funptr
    __init_symbol(&_hipMemSetAccess__funptr,"hipMemSetAccess")
    return (<hipError_t (*)(void *,unsigned long,hipMemAccessDesc *,unsigned long) nogil> _hipMemSetAccess__funptr)(ptr,size,desc,count)


cdef void* _hipMemUnmap__funptr = NULL
# 
# @brief Unmap memory allocation of a given address range.
# 
# @param [in] ptr - starting address of the range to unmap.
# @param [in] size - size of the virtual address range.
# @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
# @warning : This API is marked as beta, meaning, while this is feature complete,
# it is still open to changes and may have outstanding issues.
cdef hipError_t hipMemUnmap(void * ptr,unsigned long size) nogil:
    global _hipMemUnmap__funptr
    __init_symbol(&_hipMemUnmap__funptr,"hipMemUnmap")
    return (<hipError_t (*)(void *,unsigned long) nogil> _hipMemUnmap__funptr)(ptr,size)


cdef void* _hipGLGetDevices__funptr = NULL
cdef hipError_t hipGLGetDevices(unsigned int * pHipDeviceCount,int * pHipDevices,unsigned int hipDeviceCount,hipGLDeviceList deviceList) nogil:
    global _hipGLGetDevices__funptr
    __init_symbol(&_hipGLGetDevices__funptr,"hipGLGetDevices")
    return (<hipError_t (*)(unsigned int *,int *,unsigned int,hipGLDeviceList) nogil> _hipGLGetDevices__funptr)(pHipDeviceCount,pHipDevices,hipDeviceCount,deviceList)


cdef void* _hipGraphicsGLRegisterBuffer__funptr = NULL
cdef hipError_t hipGraphicsGLRegisterBuffer(_hipGraphicsResource ** resource,unsigned int buffer,unsigned int flags) nogil:
    global _hipGraphicsGLRegisterBuffer__funptr
    __init_symbol(&_hipGraphicsGLRegisterBuffer__funptr,"hipGraphicsGLRegisterBuffer")
    return (<hipError_t (*)(_hipGraphicsResource **,unsigned int,unsigned int) nogil> _hipGraphicsGLRegisterBuffer__funptr)(resource,buffer,flags)


cdef void* _hipGraphicsGLRegisterImage__funptr = NULL
cdef hipError_t hipGraphicsGLRegisterImage(_hipGraphicsResource ** resource,unsigned int image,unsigned int target,unsigned int flags) nogil:
    global _hipGraphicsGLRegisterImage__funptr
    __init_symbol(&_hipGraphicsGLRegisterImage__funptr,"hipGraphicsGLRegisterImage")
    return (<hipError_t (*)(_hipGraphicsResource **,unsigned int,unsigned int,unsigned int) nogil> _hipGraphicsGLRegisterImage__funptr)(resource,image,target,flags)


cdef void* _hipGraphicsMapResources__funptr = NULL
cdef hipError_t hipGraphicsMapResources(int count,hipGraphicsResource_t* resources,hipStream_t stream) nogil:
    global _hipGraphicsMapResources__funptr
    __init_symbol(&_hipGraphicsMapResources__funptr,"hipGraphicsMapResources")
    return (<hipError_t (*)(int,hipGraphicsResource_t*,hipStream_t) nogil> _hipGraphicsMapResources__funptr)(count,resources,stream)


cdef void* _hipGraphicsSubResourceGetMappedArray__funptr = NULL
cdef hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array,hipGraphicsResource_t resource,unsigned int arrayIndex,unsigned int mipLevel) nogil:
    global _hipGraphicsSubResourceGetMappedArray__funptr
    __init_symbol(&_hipGraphicsSubResourceGetMappedArray__funptr,"hipGraphicsSubResourceGetMappedArray")
    return (<hipError_t (*)(hipArray_t*,hipGraphicsResource_t,unsigned int,unsigned int) nogil> _hipGraphicsSubResourceGetMappedArray__funptr)(array,resource,arrayIndex,mipLevel)


cdef void* _hipGraphicsResourceGetMappedPointer__funptr = NULL
cdef hipError_t hipGraphicsResourceGetMappedPointer(void ** devPtr,unsigned long * size,hipGraphicsResource_t resource) nogil:
    global _hipGraphicsResourceGetMappedPointer__funptr
    __init_symbol(&_hipGraphicsResourceGetMappedPointer__funptr,"hipGraphicsResourceGetMappedPointer")
    return (<hipError_t (*)(void **,unsigned long *,hipGraphicsResource_t) nogil> _hipGraphicsResourceGetMappedPointer__funptr)(devPtr,size,resource)


cdef void* _hipGraphicsUnmapResources__funptr = NULL
cdef hipError_t hipGraphicsUnmapResources(int count,hipGraphicsResource_t* resources,hipStream_t stream) nogil:
    global _hipGraphicsUnmapResources__funptr
    __init_symbol(&_hipGraphicsUnmapResources__funptr,"hipGraphicsUnmapResources")
    return (<hipError_t (*)(int,hipGraphicsResource_t*,hipStream_t) nogil> _hipGraphicsUnmapResources__funptr)(count,resources,stream)


cdef void* _hipGraphicsUnregisterResource__funptr = NULL
cdef hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) nogil:
    global _hipGraphicsUnregisterResource__funptr
    __init_symbol(&_hipGraphicsUnregisterResource__funptr,"hipGraphicsUnregisterResource")
    return (<hipError_t (*)(hipGraphicsResource_t) nogil> _hipGraphicsUnregisterResource__funptr)(resource)


cdef void* _hipMemcpy_spt__funptr = NULL
cdef hipError_t hipMemcpy_spt(void * dst,const void * src,unsigned long sizeBytes,hipMemcpyKind kind) nogil:
    global _hipMemcpy_spt__funptr
    __init_symbol(&_hipMemcpy_spt__funptr,"hipMemcpy_spt")
    return (<hipError_t (*)(void *,const void *,unsigned long,hipMemcpyKind) nogil> _hipMemcpy_spt__funptr)(dst,src,sizeBytes,kind)


cdef void* _hipMemcpyToSymbol_spt__funptr = NULL
cdef hipError_t hipMemcpyToSymbol_spt(const void * symbol,const void * src,unsigned long sizeBytes,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipMemcpyToSymbol_spt__funptr
    __init_symbol(&_hipMemcpyToSymbol_spt__funptr,"hipMemcpyToSymbol_spt")
    return (<hipError_t (*)(const void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpyToSymbol_spt__funptr)(symbol,src,sizeBytes,offset,kind)


cdef void* _hipMemcpyFromSymbol_spt__funptr = NULL
cdef hipError_t hipMemcpyFromSymbol_spt(void * dst,const void * symbol,unsigned long sizeBytes,unsigned long offset,hipMemcpyKind kind) nogil:
    global _hipMemcpyFromSymbol_spt__funptr
    __init_symbol(&_hipMemcpyFromSymbol_spt__funptr,"hipMemcpyFromSymbol_spt")
    return (<hipError_t (*)(void *,const void *,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpyFromSymbol_spt__funptr)(dst,symbol,sizeBytes,offset,kind)


cdef void* _hipMemcpy2D_spt__funptr = NULL
cdef hipError_t hipMemcpy2D_spt(void * dst,unsigned long dpitch,const void * src,unsigned long spitch,unsigned long width,unsigned long height,hipMemcpyKind kind) nogil:
    global _hipMemcpy2D_spt__funptr
    __init_symbol(&_hipMemcpy2D_spt__funptr,"hipMemcpy2D_spt")
    return (<hipError_t (*)(void *,unsigned long,const void *,unsigned long,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpy2D_spt__funptr)(dst,dpitch,src,spitch,width,height,kind)


cdef void* _hipMemcpy2DFromArray_spt__funptr = NULL
cdef hipError_t hipMemcpy2DFromArray_spt(void * dst,unsigned long dpitch,hipArray_const_t src,unsigned long wOffset,unsigned long hOffset,unsigned long width,unsigned long height,hipMemcpyKind kind) nogil:
    global _hipMemcpy2DFromArray_spt__funptr
    __init_symbol(&_hipMemcpy2DFromArray_spt__funptr,"hipMemcpy2DFromArray_spt")
    return (<hipError_t (*)(void *,unsigned long,hipArray_const_t,unsigned long,unsigned long,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpy2DFromArray_spt__funptr)(dst,dpitch,src,wOffset,hOffset,width,height,kind)


cdef void* _hipMemcpy3D_spt__funptr = NULL
cdef hipError_t hipMemcpy3D_spt(hipMemcpy3DParms * p) nogil:
    global _hipMemcpy3D_spt__funptr
    __init_symbol(&_hipMemcpy3D_spt__funptr,"hipMemcpy3D_spt")
    return (<hipError_t (*)(hipMemcpy3DParms *) nogil> _hipMemcpy3D_spt__funptr)(p)


cdef void* _hipMemset_spt__funptr = NULL
cdef hipError_t hipMemset_spt(void * dst,int value,unsigned long sizeBytes) nogil:
    global _hipMemset_spt__funptr
    __init_symbol(&_hipMemset_spt__funptr,"hipMemset_spt")
    return (<hipError_t (*)(void *,int,unsigned long) nogil> _hipMemset_spt__funptr)(dst,value,sizeBytes)


cdef void* _hipMemsetAsync_spt__funptr = NULL
cdef hipError_t hipMemsetAsync_spt(void * dst,int value,unsigned long sizeBytes,hipStream_t stream) nogil:
    global _hipMemsetAsync_spt__funptr
    __init_symbol(&_hipMemsetAsync_spt__funptr,"hipMemsetAsync_spt")
    return (<hipError_t (*)(void *,int,unsigned long,hipStream_t) nogil> _hipMemsetAsync_spt__funptr)(dst,value,sizeBytes,stream)


cdef void* _hipMemset2D_spt__funptr = NULL
cdef hipError_t hipMemset2D_spt(void * dst,unsigned long pitch,int value,unsigned long width,unsigned long height) nogil:
    global _hipMemset2D_spt__funptr
    __init_symbol(&_hipMemset2D_spt__funptr,"hipMemset2D_spt")
    return (<hipError_t (*)(void *,unsigned long,int,unsigned long,unsigned long) nogil> _hipMemset2D_spt__funptr)(dst,pitch,value,width,height)


cdef void* _hipMemset2DAsync_spt__funptr = NULL
cdef hipError_t hipMemset2DAsync_spt(void * dst,unsigned long pitch,int value,unsigned long width,unsigned long height,hipStream_t stream) nogil:
    global _hipMemset2DAsync_spt__funptr
    __init_symbol(&_hipMemset2DAsync_spt__funptr,"hipMemset2DAsync_spt")
    return (<hipError_t (*)(void *,unsigned long,int,unsigned long,unsigned long,hipStream_t) nogil> _hipMemset2DAsync_spt__funptr)(dst,pitch,value,width,height,stream)


cdef void* _hipMemset3DAsync_spt__funptr = NULL
cdef hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent,hipStream_t stream) nogil:
    global _hipMemset3DAsync_spt__funptr
    __init_symbol(&_hipMemset3DAsync_spt__funptr,"hipMemset3DAsync_spt")
    return (<hipError_t (*)(hipPitchedPtr,int,hipExtent,hipStream_t) nogil> _hipMemset3DAsync_spt__funptr)(pitchedDevPtr,value,extent,stream)


cdef void* _hipMemset3D_spt__funptr = NULL
cdef hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent) nogil:
    global _hipMemset3D_spt__funptr
    __init_symbol(&_hipMemset3D_spt__funptr,"hipMemset3D_spt")
    return (<hipError_t (*)(hipPitchedPtr,int,hipExtent) nogil> _hipMemset3D_spt__funptr)(pitchedDevPtr,value,extent)


cdef void* _hipMemcpyAsync_spt__funptr = NULL
cdef hipError_t hipMemcpyAsync_spt(void * dst,const void * src,unsigned long sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpyAsync_spt__funptr
    __init_symbol(&_hipMemcpyAsync_spt__funptr,"hipMemcpyAsync_spt")
    return (<hipError_t (*)(void *,const void *,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpyAsync_spt__funptr)(dst,src,sizeBytes,kind,stream)


cdef void* _hipMemcpy3DAsync_spt__funptr = NULL
cdef hipError_t hipMemcpy3DAsync_spt(hipMemcpy3DParms * p,hipStream_t stream) nogil:
    global _hipMemcpy3DAsync_spt__funptr
    __init_symbol(&_hipMemcpy3DAsync_spt__funptr,"hipMemcpy3DAsync_spt")
    return (<hipError_t (*)(hipMemcpy3DParms *,hipStream_t) nogil> _hipMemcpy3DAsync_spt__funptr)(p,stream)


cdef void* _hipMemcpy2DAsync_spt__funptr = NULL
cdef hipError_t hipMemcpy2DAsync_spt(void * dst,unsigned long dpitch,const void * src,unsigned long spitch,unsigned long width,unsigned long height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpy2DAsync_spt__funptr
    __init_symbol(&_hipMemcpy2DAsync_spt__funptr,"hipMemcpy2DAsync_spt")
    return (<hipError_t (*)(void *,unsigned long,const void *,unsigned long,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpy2DAsync_spt__funptr)(dst,dpitch,src,spitch,width,height,kind,stream)


cdef void* _hipMemcpyFromSymbolAsync_spt__funptr = NULL
cdef hipError_t hipMemcpyFromSymbolAsync_spt(void * dst,const void * symbol,unsigned long sizeBytes,unsigned long offset,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpyFromSymbolAsync_spt__funptr
    __init_symbol(&_hipMemcpyFromSymbolAsync_spt__funptr,"hipMemcpyFromSymbolAsync_spt")
    return (<hipError_t (*)(void *,const void *,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpyFromSymbolAsync_spt__funptr)(dst,symbol,sizeBytes,offset,kind,stream)


cdef void* _hipMemcpyToSymbolAsync_spt__funptr = NULL
cdef hipError_t hipMemcpyToSymbolAsync_spt(const void * symbol,const void * src,unsigned long sizeBytes,unsigned long offset,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpyToSymbolAsync_spt__funptr
    __init_symbol(&_hipMemcpyToSymbolAsync_spt__funptr,"hipMemcpyToSymbolAsync_spt")
    return (<hipError_t (*)(const void *,const void *,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpyToSymbolAsync_spt__funptr)(symbol,src,sizeBytes,offset,kind,stream)


cdef void* _hipMemcpyFromArray_spt__funptr = NULL
cdef hipError_t hipMemcpyFromArray_spt(void * dst,hipArray_const_t src,unsigned long wOffsetSrc,unsigned long hOffset,unsigned long count,hipMemcpyKind kind) nogil:
    global _hipMemcpyFromArray_spt__funptr
    __init_symbol(&_hipMemcpyFromArray_spt__funptr,"hipMemcpyFromArray_spt")
    return (<hipError_t (*)(void *,hipArray_const_t,unsigned long,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpyFromArray_spt__funptr)(dst,src,wOffsetSrc,hOffset,count,kind)


cdef void* _hipMemcpy2DToArray_spt__funptr = NULL
cdef hipError_t hipMemcpy2DToArray_spt(hipArray * dst,unsigned long wOffset,unsigned long hOffset,const void * src,unsigned long spitch,unsigned long width,unsigned long height,hipMemcpyKind kind) nogil:
    global _hipMemcpy2DToArray_spt__funptr
    __init_symbol(&_hipMemcpy2DToArray_spt__funptr,"hipMemcpy2DToArray_spt")
    return (<hipError_t (*)(hipArray *,unsigned long,unsigned long,const void *,unsigned long,unsigned long,unsigned long,hipMemcpyKind) nogil> _hipMemcpy2DToArray_spt__funptr)(dst,wOffset,hOffset,src,spitch,width,height,kind)


cdef void* _hipMemcpy2DFromArrayAsync_spt__funptr = NULL
cdef hipError_t hipMemcpy2DFromArrayAsync_spt(void * dst,unsigned long dpitch,hipArray_const_t src,unsigned long wOffsetSrc,unsigned long hOffsetSrc,unsigned long width,unsigned long height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpy2DFromArrayAsync_spt__funptr
    __init_symbol(&_hipMemcpy2DFromArrayAsync_spt__funptr,"hipMemcpy2DFromArrayAsync_spt")
    return (<hipError_t (*)(void *,unsigned long,hipArray_const_t,unsigned long,unsigned long,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpy2DFromArrayAsync_spt__funptr)(dst,dpitch,src,wOffsetSrc,hOffsetSrc,width,height,kind,stream)


cdef void* _hipMemcpy2DToArrayAsync_spt__funptr = NULL
cdef hipError_t hipMemcpy2DToArrayAsync_spt(hipArray * dst,unsigned long wOffset,unsigned long hOffset,const void * src,unsigned long spitch,unsigned long width,unsigned long height,hipMemcpyKind kind,hipStream_t stream) nogil:
    global _hipMemcpy2DToArrayAsync_spt__funptr
    __init_symbol(&_hipMemcpy2DToArrayAsync_spt__funptr,"hipMemcpy2DToArrayAsync_spt")
    return (<hipError_t (*)(hipArray *,unsigned long,unsigned long,const void *,unsigned long,unsigned long,unsigned long,hipMemcpyKind,hipStream_t) nogil> _hipMemcpy2DToArrayAsync_spt__funptr)(dst,wOffset,hOffset,src,spitch,width,height,kind,stream)


cdef void* _hipStreamQuery_spt__funptr = NULL
cdef hipError_t hipStreamQuery_spt(hipStream_t stream) nogil:
    global _hipStreamQuery_spt__funptr
    __init_symbol(&_hipStreamQuery_spt__funptr,"hipStreamQuery_spt")
    return (<hipError_t (*)(hipStream_t) nogil> _hipStreamQuery_spt__funptr)(stream)


cdef void* _hipStreamSynchronize_spt__funptr = NULL
cdef hipError_t hipStreamSynchronize_spt(hipStream_t stream) nogil:
    global _hipStreamSynchronize_spt__funptr
    __init_symbol(&_hipStreamSynchronize_spt__funptr,"hipStreamSynchronize_spt")
    return (<hipError_t (*)(hipStream_t) nogil> _hipStreamSynchronize_spt__funptr)(stream)


cdef void* _hipStreamGetPriority_spt__funptr = NULL
cdef hipError_t hipStreamGetPriority_spt(hipStream_t stream,int * priority) nogil:
    global _hipStreamGetPriority_spt__funptr
    __init_symbol(&_hipStreamGetPriority_spt__funptr,"hipStreamGetPriority_spt")
    return (<hipError_t (*)(hipStream_t,int *) nogil> _hipStreamGetPriority_spt__funptr)(stream,priority)


cdef void* _hipStreamWaitEvent_spt__funptr = NULL
cdef hipError_t hipStreamWaitEvent_spt(hipStream_t stream,hipEvent_t event,unsigned int flags) nogil:
    global _hipStreamWaitEvent_spt__funptr
    __init_symbol(&_hipStreamWaitEvent_spt__funptr,"hipStreamWaitEvent_spt")
    return (<hipError_t (*)(hipStream_t,hipEvent_t,unsigned int) nogil> _hipStreamWaitEvent_spt__funptr)(stream,event,flags)


cdef void* _hipStreamGetFlags_spt__funptr = NULL
cdef hipError_t hipStreamGetFlags_spt(hipStream_t stream,unsigned int * flags) nogil:
    global _hipStreamGetFlags_spt__funptr
    __init_symbol(&_hipStreamGetFlags_spt__funptr,"hipStreamGetFlags_spt")
    return (<hipError_t (*)(hipStream_t,unsigned int *) nogil> _hipStreamGetFlags_spt__funptr)(stream,flags)


cdef void* _hipStreamAddCallback_spt__funptr = NULL
cdef hipError_t hipStreamAddCallback_spt(hipStream_t stream,hipStreamCallback_t callback,void * userData,unsigned int flags):
    global _hipStreamAddCallback_spt__funptr
    __init_symbol(&_hipStreamAddCallback_spt__funptr,"hipStreamAddCallback_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCallback_t,void *,unsigned int)> _hipStreamAddCallback_spt__funptr)(stream,callback,userData,flags)


cdef void* _hipEventRecord_spt__funptr = NULL
cdef hipError_t hipEventRecord_spt(hipEvent_t event,hipStream_t stream) nogil:
    global _hipEventRecord_spt__funptr
    __init_symbol(&_hipEventRecord_spt__funptr,"hipEventRecord_spt")
    return (<hipError_t (*)(hipEvent_t,hipStream_t) nogil> _hipEventRecord_spt__funptr)(event,stream)


cdef void* _hipLaunchCooperativeKernel_spt__funptr = NULL
cdef hipError_t hipLaunchCooperativeKernel_spt(const void * f,dim3 gridDim,dim3 blockDim,void ** kernelParams,unsigned int sharedMemBytes,hipStream_t hStream) nogil:
    global _hipLaunchCooperativeKernel_spt__funptr
    __init_symbol(&_hipLaunchCooperativeKernel_spt__funptr,"hipLaunchCooperativeKernel_spt")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,unsigned int,hipStream_t) nogil> _hipLaunchCooperativeKernel_spt__funptr)(f,gridDim,blockDim,kernelParams,sharedMemBytes,hStream)


cdef void* _hipLaunchKernel_spt__funptr = NULL
cdef hipError_t hipLaunchKernel_spt(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,unsigned long sharedMemBytes,hipStream_t stream) nogil:
    global _hipLaunchKernel_spt__funptr
    __init_symbol(&_hipLaunchKernel_spt__funptr,"hipLaunchKernel_spt")
    return (<hipError_t (*)(const void *,dim3,dim3,void **,unsigned long,hipStream_t) nogil> _hipLaunchKernel_spt__funptr)(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream)


cdef void* _hipGraphLaunch_spt__funptr = NULL
cdef hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec,hipStream_t stream) nogil:
    global _hipGraphLaunch_spt__funptr
    __init_symbol(&_hipGraphLaunch_spt__funptr,"hipGraphLaunch_spt")
    return (<hipError_t (*)(hipGraphExec_t,hipStream_t) nogil> _hipGraphLaunch_spt__funptr)(graphExec,stream)


cdef void* _hipStreamBeginCapture_spt__funptr = NULL
cdef hipError_t hipStreamBeginCapture_spt(hipStream_t stream,hipStreamCaptureMode mode) nogil:
    global _hipStreamBeginCapture_spt__funptr
    __init_symbol(&_hipStreamBeginCapture_spt__funptr,"hipStreamBeginCapture_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureMode) nogil> _hipStreamBeginCapture_spt__funptr)(stream,mode)


cdef void* _hipStreamEndCapture_spt__funptr = NULL
cdef hipError_t hipStreamEndCapture_spt(hipStream_t stream,hipGraph_t* pGraph) nogil:
    global _hipStreamEndCapture_spt__funptr
    __init_symbol(&_hipStreamEndCapture_spt__funptr,"hipStreamEndCapture_spt")
    return (<hipError_t (*)(hipStream_t,hipGraph_t*) nogil> _hipStreamEndCapture_spt__funptr)(stream,pGraph)


cdef void* _hipStreamIsCapturing_spt__funptr = NULL
cdef hipError_t hipStreamIsCapturing_spt(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus) nogil:
    global _hipStreamIsCapturing_spt__funptr
    __init_symbol(&_hipStreamIsCapturing_spt__funptr,"hipStreamIsCapturing_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *) nogil> _hipStreamIsCapturing_spt__funptr)(stream,pCaptureStatus)


cdef void* _hipStreamGetCaptureInfo_spt__funptr = NULL
cdef hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus,unsigned long long * pId) nogil:
    global _hipStreamGetCaptureInfo_spt__funptr
    __init_symbol(&_hipStreamGetCaptureInfo_spt__funptr,"hipStreamGetCaptureInfo_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *) nogil> _hipStreamGetCaptureInfo_spt__funptr)(stream,pCaptureStatus,pId)


cdef void* _hipStreamGetCaptureInfo_v2_spt__funptr = NULL
cdef hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,hipStreamCaptureStatus * captureStatus_out,unsigned long long * id_out,hipGraph_t* graph_out,hipGraphNode_t ** dependencies_out,unsigned long * numDependencies_out) nogil:
    global _hipStreamGetCaptureInfo_v2_spt__funptr
    __init_symbol(&_hipStreamGetCaptureInfo_v2_spt__funptr,"hipStreamGetCaptureInfo_v2_spt")
    return (<hipError_t (*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *,hipGraph_t*,hipGraphNode_t **,unsigned long *) nogil> _hipStreamGetCaptureInfo_v2_spt__funptr)(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out)


cdef void* _hipLaunchHostFunc_spt__funptr = NULL
cdef hipError_t hipLaunchHostFunc_spt(hipStream_t stream,hipHostFn_t fn,void * userData):
    global _hipLaunchHostFunc_spt__funptr
    __init_symbol(&_hipLaunchHostFunc_spt__funptr,"hipLaunchHostFunc_spt")
    return (<hipError_t (*)(hipStream_t,hipHostFn_t,void *)> _hipLaunchHostFunc_spt__funptr)(stream,fn,userData)
