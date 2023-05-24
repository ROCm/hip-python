# AMD_COPYRIGHT

cimport hip.chip

from hip.chip cimport HIP_TRSA_OVERRIDE_FORMAT as CU_TRSA_OVERRIDE_FORMAT
from hip.chip cimport HIP_TRSF_READ_AS_INTEGER as CU_TRSF_READ_AS_INTEGER
from hip.chip cimport HIP_TRSF_NORMALIZED_COORDINATES as CU_TRSF_NORMALIZED_COORDINATES
from hip.chip cimport HIP_TRSF_SRGB as CU_TRSF_SRGB
from hip.chip cimport hipTextureType1D as cudaTextureType1D
from hip.chip cimport hipTextureType2D as cudaTextureType2D
from hip.chip cimport hipTextureType3D as cudaTextureType3D
from hip.chip cimport hipTextureTypeCubemap as cudaTextureTypeCubemap
from hip.chip cimport hipTextureType1DLayered as cudaTextureType1DLayered
from hip.chip cimport hipTextureType2DLayered as cudaTextureType2DLayered
from hip.chip cimport hipTextureTypeCubemapLayered as cudaTextureTypeCubemapLayered
from hip.chip cimport HIP_LAUNCH_PARAM_BUFFER_POINTER as CU_LAUNCH_PARAM_BUFFER_POINTER
from hip.chip cimport HIP_LAUNCH_PARAM_BUFFER_SIZE as CU_LAUNCH_PARAM_BUFFER_SIZE
from hip.chip cimport HIP_LAUNCH_PARAM_END as CU_LAUNCH_PARAM_END
from hip.chip cimport hipIpcMemLazyEnablePeerAccess as CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
from hip.chip cimport hipIpcMemLazyEnablePeerAccess as cudaIpcMemLazyEnablePeerAccess
from hip.chip cimport HIP_IPC_HANDLE_SIZE as CUDA_IPC_HANDLE_SIZE
from hip.chip cimport HIP_IPC_HANDLE_SIZE as CU_IPC_HANDLE_SIZE
from hip.chip cimport hipStreamDefault as CU_STREAM_DEFAULT
from hip.chip cimport hipStreamDefault as cudaStreamDefault
from hip.chip cimport hipStreamNonBlocking as CU_STREAM_NON_BLOCKING
from hip.chip cimport hipStreamNonBlocking as cudaStreamNonBlocking
from hip.chip cimport hipEventDefault as CU_EVENT_DEFAULT
from hip.chip cimport hipEventDefault as cudaEventDefault
from hip.chip cimport hipEventBlockingSync as CU_EVENT_BLOCKING_SYNC
from hip.chip cimport hipEventBlockingSync as cudaEventBlockingSync
from hip.chip cimport hipEventDisableTiming as CU_EVENT_DISABLE_TIMING
from hip.chip cimport hipEventDisableTiming as cudaEventDisableTiming
from hip.chip cimport hipEventInterprocess as CU_EVENT_INTERPROCESS
from hip.chip cimport hipEventInterprocess as cudaEventInterprocess
from hip.chip cimport hipHostMallocDefault as cudaHostAllocDefault
from hip.chip cimport hipHostMallocPortable as CU_MEMHOSTALLOC_PORTABLE
from hip.chip cimport hipHostMallocPortable as cudaHostAllocPortable
from hip.chip cimport hipHostMallocMapped as CU_MEMHOSTALLOC_DEVICEMAP
from hip.chip cimport hipHostMallocMapped as cudaHostAllocMapped
from hip.chip cimport hipHostMallocWriteCombined as CU_MEMHOSTALLOC_WRITECOMBINED
from hip.chip cimport hipHostMallocWriteCombined as cudaHostAllocWriteCombined
from hip.chip cimport hipMemAttachGlobal as CU_MEM_ATTACH_GLOBAL
from hip.chip cimport hipMemAttachGlobal as cudaMemAttachGlobal
from hip.chip cimport hipMemAttachHost as CU_MEM_ATTACH_HOST
from hip.chip cimport hipMemAttachHost as cudaMemAttachHost
from hip.chip cimport hipMemAttachSingle as CU_MEM_ATTACH_SINGLE
from hip.chip cimport hipMemAttachSingle as cudaMemAttachSingle
from hip.chip cimport hipHostRegisterDefault as cudaHostRegisterDefault
from hip.chip cimport hipHostRegisterPortable as CU_MEMHOSTREGISTER_PORTABLE
from hip.chip cimport hipHostRegisterPortable as cudaHostRegisterPortable
from hip.chip cimport hipHostRegisterMapped as CU_MEMHOSTREGISTER_DEVICEMAP
from hip.chip cimport hipHostRegisterMapped as cudaHostRegisterMapped
from hip.chip cimport hipHostRegisterIoMemory as CU_MEMHOSTREGISTER_IOMEMORY
from hip.chip cimport hipHostRegisterIoMemory as cudaHostRegisterIoMemory
from hip.chip cimport hipDeviceScheduleAuto as CU_CTX_SCHED_AUTO
from hip.chip cimport hipDeviceScheduleAuto as cudaDeviceScheduleAuto
from hip.chip cimport hipDeviceScheduleSpin as CU_CTX_SCHED_SPIN
from hip.chip cimport hipDeviceScheduleSpin as cudaDeviceScheduleSpin
from hip.chip cimport hipDeviceScheduleYield as CU_CTX_SCHED_YIELD
from hip.chip cimport hipDeviceScheduleYield as cudaDeviceScheduleYield
from hip.chip cimport hipDeviceScheduleBlockingSync as CU_CTX_BLOCKING_SYNC
from hip.chip cimport hipDeviceScheduleBlockingSync as CU_CTX_SCHED_BLOCKING_SYNC
from hip.chip cimport hipDeviceScheduleBlockingSync as cudaDeviceBlockingSync
from hip.chip cimport hipDeviceScheduleBlockingSync as cudaDeviceScheduleBlockingSync
from hip.chip cimport hipDeviceScheduleMask as CU_CTX_SCHED_MASK
from hip.chip cimport hipDeviceScheduleMask as cudaDeviceScheduleMask
from hip.chip cimport hipDeviceMapHost as CU_CTX_MAP_HOST
from hip.chip cimport hipDeviceMapHost as cudaDeviceMapHost
from hip.chip cimport hipDeviceLmemResizeToMax as CU_CTX_LMEM_RESIZE_TO_MAX
from hip.chip cimport hipDeviceLmemResizeToMax as cudaDeviceLmemResizeToMax
from hip.chip cimport hipArrayDefault as cudaArrayDefault
from hip.chip cimport hipArrayLayered as CUDA_ARRAY3D_LAYERED
from hip.chip cimport hipArrayLayered as cudaArrayLayered
from hip.chip cimport hipArraySurfaceLoadStore as CUDA_ARRAY3D_SURFACE_LDST
from hip.chip cimport hipArraySurfaceLoadStore as cudaArraySurfaceLoadStore
from hip.chip cimport hipArrayCubemap as CUDA_ARRAY3D_CUBEMAP
from hip.chip cimport hipArrayCubemap as cudaArrayCubemap
from hip.chip cimport hipArrayTextureGather as CUDA_ARRAY3D_TEXTURE_GATHER
from hip.chip cimport hipArrayTextureGather as cudaArrayTextureGather
from hip.chip cimport hipOccupancyDefault as CU_OCCUPANCY_DEFAULT
from hip.chip cimport hipOccupancyDefault as cudaOccupancyDefault
from hip.chip cimport hipCooperativeLaunchMultiDeviceNoPreSync as CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC
from hip.chip cimport hipCooperativeLaunchMultiDeviceNoPreSync as cudaCooperativeLaunchMultiDeviceNoPreSync
from hip.chip cimport hipCooperativeLaunchMultiDeviceNoPostSync as CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC
from hip.chip cimport hipCooperativeLaunchMultiDeviceNoPostSync as cudaCooperativeLaunchMultiDeviceNoPostSync
from hip.chip cimport hipCpuDeviceId as CU_DEVICE_CPU
from hip.chip cimport hipCpuDeviceId as cudaCpuDeviceId
from hip.chip cimport hipInvalidDeviceId as CU_DEVICE_INVALID
from hip.chip cimport hipInvalidDeviceId as cudaInvalidDeviceId
from hip.chip cimport hipStreamWaitValueGte as CU_STREAM_WAIT_VALUE_GEQ
from hip.chip cimport hipStreamWaitValueEq as CU_STREAM_WAIT_VALUE_EQ
from hip.chip cimport hipStreamWaitValueAnd as CU_STREAM_WAIT_VALUE_AND
from hip.chip cimport hipStreamWaitValueNor as CU_STREAM_WAIT_VALUE_NOR
cdef enum enum_1:
    HIP_SUCCESS = hip.chip.HIP_SUCCESS
    HIP_ERROR_INVALID_VALUE = hip.chip.HIP_ERROR_INVALID_VALUE
    HIP_ERROR_NOT_INITIALIZED = hip.chip.HIP_ERROR_NOT_INITIALIZED
    HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = hip.chip.HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
from hip.chip cimport hipUUID_t as CUuuid_st
from hip.chip cimport hipDeviceProp_t as cudaDeviceProp
cdef enum CUmemorytype:
    CU_MEMORYTYPE_HOST = hip.chip.hipMemoryTypeHost
    cudaMemoryTypeHost = hip.chip.hipMemoryTypeHost
    hipMemoryTypeHost = hip.chip.hipMemoryTypeHost
    CU_MEMORYTYPE_DEVICE = hip.chip.hipMemoryTypeDevice
    cudaMemoryTypeDevice = hip.chip.hipMemoryTypeDevice
    hipMemoryTypeDevice = hip.chip.hipMemoryTypeDevice
    CU_MEMORYTYPE_ARRAY = hip.chip.hipMemoryTypeArray
    hipMemoryTypeArray = hip.chip.hipMemoryTypeArray
    CU_MEMORYTYPE_UNIFIED = hip.chip.hipMemoryTypeUnified
    hipMemoryTypeUnified = hip.chip.hipMemoryTypeUnified
    cudaMemoryTypeManaged = hip.chip.hipMemoryTypeManaged
    hipMemoryTypeManaged = hip.chip.hipMemoryTypeManaged
ctypedef CUmemorytype CUmemorytype_enum
ctypedef CUmemorytype cudaMemoryType
from hip.chip cimport hipPointerAttribute_t as cudaPointerAttributes
cdef enum CUresult:
    CUDA_SUCCESS = hip.chip.hipSuccess
    cudaSuccess = hip.chip.hipSuccess
    hipSuccess = hip.chip.hipSuccess
    CUDA_ERROR_INVALID_VALUE = hip.chip.hipErrorInvalidValue
    cudaErrorInvalidValue = hip.chip.hipErrorInvalidValue
    hipErrorInvalidValue = hip.chip.hipErrorInvalidValue
    CUDA_ERROR_OUT_OF_MEMORY = hip.chip.hipErrorOutOfMemory
    cudaErrorMemoryAllocation = hip.chip.hipErrorOutOfMemory
    hipErrorOutOfMemory = hip.chip.hipErrorOutOfMemory
    hipErrorMemoryAllocation = hip.chip.hipErrorMemoryAllocation
    CUDA_ERROR_NOT_INITIALIZED = hip.chip.hipErrorNotInitialized
    cudaErrorInitializationError = hip.chip.hipErrorNotInitialized
    hipErrorNotInitialized = hip.chip.hipErrorNotInitialized
    hipErrorInitializationError = hip.chip.hipErrorInitializationError
    CUDA_ERROR_DEINITIALIZED = hip.chip.hipErrorDeinitialized
    cudaErrorCudartUnloading = hip.chip.hipErrorDeinitialized
    hipErrorDeinitialized = hip.chip.hipErrorDeinitialized
    CUDA_ERROR_PROFILER_DISABLED = hip.chip.hipErrorProfilerDisabled
    cudaErrorProfilerDisabled = hip.chip.hipErrorProfilerDisabled
    hipErrorProfilerDisabled = hip.chip.hipErrorProfilerDisabled
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = hip.chip.hipErrorProfilerNotInitialized
    cudaErrorProfilerNotInitialized = hip.chip.hipErrorProfilerNotInitialized
    hipErrorProfilerNotInitialized = hip.chip.hipErrorProfilerNotInitialized
    CUDA_ERROR_PROFILER_ALREADY_STARTED = hip.chip.hipErrorProfilerAlreadyStarted
    cudaErrorProfilerAlreadyStarted = hip.chip.hipErrorProfilerAlreadyStarted
    hipErrorProfilerAlreadyStarted = hip.chip.hipErrorProfilerAlreadyStarted
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = hip.chip.hipErrorProfilerAlreadyStopped
    cudaErrorProfilerAlreadyStopped = hip.chip.hipErrorProfilerAlreadyStopped
    hipErrorProfilerAlreadyStopped = hip.chip.hipErrorProfilerAlreadyStopped
    cudaErrorInvalidConfiguration = hip.chip.hipErrorInvalidConfiguration
    hipErrorInvalidConfiguration = hip.chip.hipErrorInvalidConfiguration
    cudaErrorInvalidPitchValue = hip.chip.hipErrorInvalidPitchValue
    hipErrorInvalidPitchValue = hip.chip.hipErrorInvalidPitchValue
    cudaErrorInvalidSymbol = hip.chip.hipErrorInvalidSymbol
    hipErrorInvalidSymbol = hip.chip.hipErrorInvalidSymbol
    cudaErrorInvalidDevicePointer = hip.chip.hipErrorInvalidDevicePointer
    hipErrorInvalidDevicePointer = hip.chip.hipErrorInvalidDevicePointer
    cudaErrorInvalidMemcpyDirection = hip.chip.hipErrorInvalidMemcpyDirection
    hipErrorInvalidMemcpyDirection = hip.chip.hipErrorInvalidMemcpyDirection
    cudaErrorInsufficientDriver = hip.chip.hipErrorInsufficientDriver
    hipErrorInsufficientDriver = hip.chip.hipErrorInsufficientDriver
    cudaErrorMissingConfiguration = hip.chip.hipErrorMissingConfiguration
    hipErrorMissingConfiguration = hip.chip.hipErrorMissingConfiguration
    cudaErrorPriorLaunchFailure = hip.chip.hipErrorPriorLaunchFailure
    hipErrorPriorLaunchFailure = hip.chip.hipErrorPriorLaunchFailure
    cudaErrorInvalidDeviceFunction = hip.chip.hipErrorInvalidDeviceFunction
    hipErrorInvalidDeviceFunction = hip.chip.hipErrorInvalidDeviceFunction
    CUDA_ERROR_NO_DEVICE = hip.chip.hipErrorNoDevice
    cudaErrorNoDevice = hip.chip.hipErrorNoDevice
    hipErrorNoDevice = hip.chip.hipErrorNoDevice
    CUDA_ERROR_INVALID_DEVICE = hip.chip.hipErrorInvalidDevice
    cudaErrorInvalidDevice = hip.chip.hipErrorInvalidDevice
    hipErrorInvalidDevice = hip.chip.hipErrorInvalidDevice
    CUDA_ERROR_INVALID_IMAGE = hip.chip.hipErrorInvalidImage
    cudaErrorInvalidKernelImage = hip.chip.hipErrorInvalidImage
    hipErrorInvalidImage = hip.chip.hipErrorInvalidImage
    CUDA_ERROR_INVALID_CONTEXT = hip.chip.hipErrorInvalidContext
    cudaErrorDeviceUninitialized = hip.chip.hipErrorInvalidContext
    hipErrorInvalidContext = hip.chip.hipErrorInvalidContext
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = hip.chip.hipErrorContextAlreadyCurrent
    hipErrorContextAlreadyCurrent = hip.chip.hipErrorContextAlreadyCurrent
    CUDA_ERROR_MAP_FAILED = hip.chip.hipErrorMapFailed
    cudaErrorMapBufferObjectFailed = hip.chip.hipErrorMapFailed
    hipErrorMapFailed = hip.chip.hipErrorMapFailed
    hipErrorMapBufferObjectFailed = hip.chip.hipErrorMapBufferObjectFailed
    CUDA_ERROR_UNMAP_FAILED = hip.chip.hipErrorUnmapFailed
    cudaErrorUnmapBufferObjectFailed = hip.chip.hipErrorUnmapFailed
    hipErrorUnmapFailed = hip.chip.hipErrorUnmapFailed
    CUDA_ERROR_ARRAY_IS_MAPPED = hip.chip.hipErrorArrayIsMapped
    cudaErrorArrayIsMapped = hip.chip.hipErrorArrayIsMapped
    hipErrorArrayIsMapped = hip.chip.hipErrorArrayIsMapped
    CUDA_ERROR_ALREADY_MAPPED = hip.chip.hipErrorAlreadyMapped
    cudaErrorAlreadyMapped = hip.chip.hipErrorAlreadyMapped
    hipErrorAlreadyMapped = hip.chip.hipErrorAlreadyMapped
    CUDA_ERROR_NO_BINARY_FOR_GPU = hip.chip.hipErrorNoBinaryForGpu
    cudaErrorNoKernelImageForDevice = hip.chip.hipErrorNoBinaryForGpu
    hipErrorNoBinaryForGpu = hip.chip.hipErrorNoBinaryForGpu
    CUDA_ERROR_ALREADY_ACQUIRED = hip.chip.hipErrorAlreadyAcquired
    cudaErrorAlreadyAcquired = hip.chip.hipErrorAlreadyAcquired
    hipErrorAlreadyAcquired = hip.chip.hipErrorAlreadyAcquired
    CUDA_ERROR_NOT_MAPPED = hip.chip.hipErrorNotMapped
    cudaErrorNotMapped = hip.chip.hipErrorNotMapped
    hipErrorNotMapped = hip.chip.hipErrorNotMapped
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = hip.chip.hipErrorNotMappedAsArray
    cudaErrorNotMappedAsArray = hip.chip.hipErrorNotMappedAsArray
    hipErrorNotMappedAsArray = hip.chip.hipErrorNotMappedAsArray
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = hip.chip.hipErrorNotMappedAsPointer
    cudaErrorNotMappedAsPointer = hip.chip.hipErrorNotMappedAsPointer
    hipErrorNotMappedAsPointer = hip.chip.hipErrorNotMappedAsPointer
    CUDA_ERROR_ECC_UNCORRECTABLE = hip.chip.hipErrorECCNotCorrectable
    cudaErrorECCUncorrectable = hip.chip.hipErrorECCNotCorrectable
    hipErrorECCNotCorrectable = hip.chip.hipErrorECCNotCorrectable
    CUDA_ERROR_UNSUPPORTED_LIMIT = hip.chip.hipErrorUnsupportedLimit
    cudaErrorUnsupportedLimit = hip.chip.hipErrorUnsupportedLimit
    hipErrorUnsupportedLimit = hip.chip.hipErrorUnsupportedLimit
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = hip.chip.hipErrorContextAlreadyInUse
    cudaErrorDeviceAlreadyInUse = hip.chip.hipErrorContextAlreadyInUse
    hipErrorContextAlreadyInUse = hip.chip.hipErrorContextAlreadyInUse
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = hip.chip.hipErrorPeerAccessUnsupported
    cudaErrorPeerAccessUnsupported = hip.chip.hipErrorPeerAccessUnsupported
    hipErrorPeerAccessUnsupported = hip.chip.hipErrorPeerAccessUnsupported
    CUDA_ERROR_INVALID_PTX = hip.chip.hipErrorInvalidKernelFile
    cudaErrorInvalidPtx = hip.chip.hipErrorInvalidKernelFile
    hipErrorInvalidKernelFile = hip.chip.hipErrorInvalidKernelFile
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = hip.chip.hipErrorInvalidGraphicsContext
    cudaErrorInvalidGraphicsContext = hip.chip.hipErrorInvalidGraphicsContext
    hipErrorInvalidGraphicsContext = hip.chip.hipErrorInvalidGraphicsContext
    CUDA_ERROR_INVALID_SOURCE = hip.chip.hipErrorInvalidSource
    cudaErrorInvalidSource = hip.chip.hipErrorInvalidSource
    hipErrorInvalidSource = hip.chip.hipErrorInvalidSource
    CUDA_ERROR_FILE_NOT_FOUND = hip.chip.hipErrorFileNotFound
    cudaErrorFileNotFound = hip.chip.hipErrorFileNotFound
    hipErrorFileNotFound = hip.chip.hipErrorFileNotFound
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = hip.chip.hipErrorSharedObjectSymbolNotFound
    cudaErrorSharedObjectSymbolNotFound = hip.chip.hipErrorSharedObjectSymbolNotFound
    hipErrorSharedObjectSymbolNotFound = hip.chip.hipErrorSharedObjectSymbolNotFound
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = hip.chip.hipErrorSharedObjectInitFailed
    cudaErrorSharedObjectInitFailed = hip.chip.hipErrorSharedObjectInitFailed
    hipErrorSharedObjectInitFailed = hip.chip.hipErrorSharedObjectInitFailed
    CUDA_ERROR_OPERATING_SYSTEM = hip.chip.hipErrorOperatingSystem
    cudaErrorOperatingSystem = hip.chip.hipErrorOperatingSystem
    hipErrorOperatingSystem = hip.chip.hipErrorOperatingSystem
    CUDA_ERROR_INVALID_HANDLE = hip.chip.hipErrorInvalidHandle
    cudaErrorInvalidResourceHandle = hip.chip.hipErrorInvalidHandle
    hipErrorInvalidHandle = hip.chip.hipErrorInvalidHandle
    hipErrorInvalidResourceHandle = hip.chip.hipErrorInvalidResourceHandle
    CUDA_ERROR_ILLEGAL_STATE = hip.chip.hipErrorIllegalState
    cudaErrorIllegalState = hip.chip.hipErrorIllegalState
    hipErrorIllegalState = hip.chip.hipErrorIllegalState
    CUDA_ERROR_NOT_FOUND = hip.chip.hipErrorNotFound
    cudaErrorSymbolNotFound = hip.chip.hipErrorNotFound
    hipErrorNotFound = hip.chip.hipErrorNotFound
    CUDA_ERROR_NOT_READY = hip.chip.hipErrorNotReady
    cudaErrorNotReady = hip.chip.hipErrorNotReady
    hipErrorNotReady = hip.chip.hipErrorNotReady
    CUDA_ERROR_ILLEGAL_ADDRESS = hip.chip.hipErrorIllegalAddress
    cudaErrorIllegalAddress = hip.chip.hipErrorIllegalAddress
    hipErrorIllegalAddress = hip.chip.hipErrorIllegalAddress
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = hip.chip.hipErrorLaunchOutOfResources
    cudaErrorLaunchOutOfResources = hip.chip.hipErrorLaunchOutOfResources
    hipErrorLaunchOutOfResources = hip.chip.hipErrorLaunchOutOfResources
    CUDA_ERROR_LAUNCH_TIMEOUT = hip.chip.hipErrorLaunchTimeOut
    cudaErrorLaunchTimeout = hip.chip.hipErrorLaunchTimeOut
    hipErrorLaunchTimeOut = hip.chip.hipErrorLaunchTimeOut
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = hip.chip.hipErrorPeerAccessAlreadyEnabled
    cudaErrorPeerAccessAlreadyEnabled = hip.chip.hipErrorPeerAccessAlreadyEnabled
    hipErrorPeerAccessAlreadyEnabled = hip.chip.hipErrorPeerAccessAlreadyEnabled
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = hip.chip.hipErrorPeerAccessNotEnabled
    cudaErrorPeerAccessNotEnabled = hip.chip.hipErrorPeerAccessNotEnabled
    hipErrorPeerAccessNotEnabled = hip.chip.hipErrorPeerAccessNotEnabled
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = hip.chip.hipErrorSetOnActiveProcess
    cudaErrorSetOnActiveProcess = hip.chip.hipErrorSetOnActiveProcess
    hipErrorSetOnActiveProcess = hip.chip.hipErrorSetOnActiveProcess
    CUDA_ERROR_CONTEXT_IS_DESTROYED = hip.chip.hipErrorContextIsDestroyed
    cudaErrorContextIsDestroyed = hip.chip.hipErrorContextIsDestroyed
    hipErrorContextIsDestroyed = hip.chip.hipErrorContextIsDestroyed
    CUDA_ERROR_ASSERT = hip.chip.hipErrorAssert
    cudaErrorAssert = hip.chip.hipErrorAssert
    hipErrorAssert = hip.chip.hipErrorAssert
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = hip.chip.hipErrorHostMemoryAlreadyRegistered
    cudaErrorHostMemoryAlreadyRegistered = hip.chip.hipErrorHostMemoryAlreadyRegistered
    hipErrorHostMemoryAlreadyRegistered = hip.chip.hipErrorHostMemoryAlreadyRegistered
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = hip.chip.hipErrorHostMemoryNotRegistered
    cudaErrorHostMemoryNotRegistered = hip.chip.hipErrorHostMemoryNotRegistered
    hipErrorHostMemoryNotRegistered = hip.chip.hipErrorHostMemoryNotRegistered
    CUDA_ERROR_LAUNCH_FAILED = hip.chip.hipErrorLaunchFailure
    cudaErrorLaunchFailure = hip.chip.hipErrorLaunchFailure
    hipErrorLaunchFailure = hip.chip.hipErrorLaunchFailure
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = hip.chip.hipErrorCooperativeLaunchTooLarge
    cudaErrorCooperativeLaunchTooLarge = hip.chip.hipErrorCooperativeLaunchTooLarge
    hipErrorCooperativeLaunchTooLarge = hip.chip.hipErrorCooperativeLaunchTooLarge
    CUDA_ERROR_NOT_SUPPORTED = hip.chip.hipErrorNotSupported
    cudaErrorNotSupported = hip.chip.hipErrorNotSupported
    hipErrorNotSupported = hip.chip.hipErrorNotSupported
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = hip.chip.hipErrorStreamCaptureUnsupported
    cudaErrorStreamCaptureUnsupported = hip.chip.hipErrorStreamCaptureUnsupported
    hipErrorStreamCaptureUnsupported = hip.chip.hipErrorStreamCaptureUnsupported
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = hip.chip.hipErrorStreamCaptureInvalidated
    cudaErrorStreamCaptureInvalidated = hip.chip.hipErrorStreamCaptureInvalidated
    hipErrorStreamCaptureInvalidated = hip.chip.hipErrorStreamCaptureInvalidated
    CUDA_ERROR_STREAM_CAPTURE_MERGE = hip.chip.hipErrorStreamCaptureMerge
    cudaErrorStreamCaptureMerge = hip.chip.hipErrorStreamCaptureMerge
    hipErrorStreamCaptureMerge = hip.chip.hipErrorStreamCaptureMerge
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = hip.chip.hipErrorStreamCaptureUnmatched
    cudaErrorStreamCaptureUnmatched = hip.chip.hipErrorStreamCaptureUnmatched
    hipErrorStreamCaptureUnmatched = hip.chip.hipErrorStreamCaptureUnmatched
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = hip.chip.hipErrorStreamCaptureUnjoined
    cudaErrorStreamCaptureUnjoined = hip.chip.hipErrorStreamCaptureUnjoined
    hipErrorStreamCaptureUnjoined = hip.chip.hipErrorStreamCaptureUnjoined
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = hip.chip.hipErrorStreamCaptureIsolation
    cudaErrorStreamCaptureIsolation = hip.chip.hipErrorStreamCaptureIsolation
    hipErrorStreamCaptureIsolation = hip.chip.hipErrorStreamCaptureIsolation
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = hip.chip.hipErrorStreamCaptureImplicit
    cudaErrorStreamCaptureImplicit = hip.chip.hipErrorStreamCaptureImplicit
    hipErrorStreamCaptureImplicit = hip.chip.hipErrorStreamCaptureImplicit
    CUDA_ERROR_CAPTURED_EVENT = hip.chip.hipErrorCapturedEvent
    cudaErrorCapturedEvent = hip.chip.hipErrorCapturedEvent
    hipErrorCapturedEvent = hip.chip.hipErrorCapturedEvent
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = hip.chip.hipErrorStreamCaptureWrongThread
    cudaErrorStreamCaptureWrongThread = hip.chip.hipErrorStreamCaptureWrongThread
    hipErrorStreamCaptureWrongThread = hip.chip.hipErrorStreamCaptureWrongThread
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = hip.chip.hipErrorGraphExecUpdateFailure
    cudaErrorGraphExecUpdateFailure = hip.chip.hipErrorGraphExecUpdateFailure
    hipErrorGraphExecUpdateFailure = hip.chip.hipErrorGraphExecUpdateFailure
    CUDA_ERROR_UNKNOWN = hip.chip.hipErrorUnknown
    cudaErrorUnknown = hip.chip.hipErrorUnknown
    hipErrorUnknown = hip.chip.hipErrorUnknown
    hipErrorRuntimeMemory = hip.chip.hipErrorRuntimeMemory
    hipErrorRuntimeOther = hip.chip.hipErrorRuntimeOther
    hipErrorTbd = hip.chip.hipErrorTbd
ctypedef CUresult cudaError
ctypedef CUresult cudaError_enum
ctypedef CUresult cudaError_t
cdef enum CUdevice_attribute:
    hipDeviceAttributeCudaCompatibleBegin = hip.chip.hipDeviceAttributeCudaCompatibleBegin
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = hip.chip.hipDeviceAttributeEccEnabled
    cudaDevAttrEccEnabled = hip.chip.hipDeviceAttributeEccEnabled
    hipDeviceAttributeEccEnabled = hip.chip.hipDeviceAttributeEccEnabled
    hipDeviceAttributeAccessPolicyMaxWindowSize = hip.chip.hipDeviceAttributeAccessPolicyMaxWindowSize
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = hip.chip.hipDeviceAttributeAsyncEngineCount
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = hip.chip.hipDeviceAttributeAsyncEngineCount
    cudaDevAttrAsyncEngineCount = hip.chip.hipDeviceAttributeAsyncEngineCount
    cudaDevAttrGpuOverlap = hip.chip.hipDeviceAttributeAsyncEngineCount
    hipDeviceAttributeAsyncEngineCount = hip.chip.hipDeviceAttributeAsyncEngineCount
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = hip.chip.hipDeviceAttributeCanMapHostMemory
    cudaDevAttrCanMapHostMemory = hip.chip.hipDeviceAttributeCanMapHostMemory
    hipDeviceAttributeCanMapHostMemory = hip.chip.hipDeviceAttributeCanMapHostMemory
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = hip.chip.hipDeviceAttributeCanUseHostPointerForRegisteredMem
    cudaDevAttrCanUseHostPointerForRegisteredMem = hip.chip.hipDeviceAttributeCanUseHostPointerForRegisteredMem
    hipDeviceAttributeCanUseHostPointerForRegisteredMem = hip.chip.hipDeviceAttributeCanUseHostPointerForRegisteredMem
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = hip.chip.hipDeviceAttributeClockRate
    cudaDevAttrClockRate = hip.chip.hipDeviceAttributeClockRate
    hipDeviceAttributeClockRate = hip.chip.hipDeviceAttributeClockRate
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = hip.chip.hipDeviceAttributeComputeMode
    cudaDevAttrComputeMode = hip.chip.hipDeviceAttributeComputeMode
    hipDeviceAttributeComputeMode = hip.chip.hipDeviceAttributeComputeMode
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = hip.chip.hipDeviceAttributeComputePreemptionSupported
    cudaDevAttrComputePreemptionSupported = hip.chip.hipDeviceAttributeComputePreemptionSupported
    hipDeviceAttributeComputePreemptionSupported = hip.chip.hipDeviceAttributeComputePreemptionSupported
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = hip.chip.hipDeviceAttributeConcurrentKernels
    cudaDevAttrConcurrentKernels = hip.chip.hipDeviceAttributeConcurrentKernels
    hipDeviceAttributeConcurrentKernels = hip.chip.hipDeviceAttributeConcurrentKernels
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = hip.chip.hipDeviceAttributeConcurrentManagedAccess
    cudaDevAttrConcurrentManagedAccess = hip.chip.hipDeviceAttributeConcurrentManagedAccess
    hipDeviceAttributeConcurrentManagedAccess = hip.chip.hipDeviceAttributeConcurrentManagedAccess
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = hip.chip.hipDeviceAttributeCooperativeLaunch
    cudaDevAttrCooperativeLaunch = hip.chip.hipDeviceAttributeCooperativeLaunch
    hipDeviceAttributeCooperativeLaunch = hip.chip.hipDeviceAttributeCooperativeLaunch
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = hip.chip.hipDeviceAttributeCooperativeMultiDeviceLaunch
    cudaDevAttrCooperativeMultiDeviceLaunch = hip.chip.hipDeviceAttributeCooperativeMultiDeviceLaunch
    hipDeviceAttributeCooperativeMultiDeviceLaunch = hip.chip.hipDeviceAttributeCooperativeMultiDeviceLaunch
    hipDeviceAttributeDeviceOverlap = hip.chip.hipDeviceAttributeDeviceOverlap
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = hip.chip.hipDeviceAttributeDirectManagedMemAccessFromHost
    cudaDevAttrDirectManagedMemAccessFromHost = hip.chip.hipDeviceAttributeDirectManagedMemAccessFromHost
    hipDeviceAttributeDirectManagedMemAccessFromHost = hip.chip.hipDeviceAttributeDirectManagedMemAccessFromHost
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = hip.chip.hipDeviceAttributeGlobalL1CacheSupported
    cudaDevAttrGlobalL1CacheSupported = hip.chip.hipDeviceAttributeGlobalL1CacheSupported
    hipDeviceAttributeGlobalL1CacheSupported = hip.chip.hipDeviceAttributeGlobalL1CacheSupported
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = hip.chip.hipDeviceAttributeHostNativeAtomicSupported
    cudaDevAttrHostNativeAtomicSupported = hip.chip.hipDeviceAttributeHostNativeAtomicSupported
    hipDeviceAttributeHostNativeAtomicSupported = hip.chip.hipDeviceAttributeHostNativeAtomicSupported
    CU_DEVICE_ATTRIBUTE_INTEGRATED = hip.chip.hipDeviceAttributeIntegrated
    cudaDevAttrIntegrated = hip.chip.hipDeviceAttributeIntegrated
    hipDeviceAttributeIntegrated = hip.chip.hipDeviceAttributeIntegrated
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = hip.chip.hipDeviceAttributeIsMultiGpuBoard
    cudaDevAttrIsMultiGpuBoard = hip.chip.hipDeviceAttributeIsMultiGpuBoard
    hipDeviceAttributeIsMultiGpuBoard = hip.chip.hipDeviceAttributeIsMultiGpuBoard
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = hip.chip.hipDeviceAttributeKernelExecTimeout
    cudaDevAttrKernelExecTimeout = hip.chip.hipDeviceAttributeKernelExecTimeout
    hipDeviceAttributeKernelExecTimeout = hip.chip.hipDeviceAttributeKernelExecTimeout
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = hip.chip.hipDeviceAttributeL2CacheSize
    cudaDevAttrL2CacheSize = hip.chip.hipDeviceAttributeL2CacheSize
    hipDeviceAttributeL2CacheSize = hip.chip.hipDeviceAttributeL2CacheSize
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = hip.chip.hipDeviceAttributeLocalL1CacheSupported
    cudaDevAttrLocalL1CacheSupported = hip.chip.hipDeviceAttributeLocalL1CacheSupported
    hipDeviceAttributeLocalL1CacheSupported = hip.chip.hipDeviceAttributeLocalL1CacheSupported
    hipDeviceAttributeLuid = hip.chip.hipDeviceAttributeLuid
    hipDeviceAttributeLuidDeviceNodeMask = hip.chip.hipDeviceAttributeLuidDeviceNodeMask
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = hip.chip.hipDeviceAttributeComputeCapabilityMajor
    cudaDevAttrComputeCapabilityMajor = hip.chip.hipDeviceAttributeComputeCapabilityMajor
    hipDeviceAttributeComputeCapabilityMajor = hip.chip.hipDeviceAttributeComputeCapabilityMajor
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = hip.chip.hipDeviceAttributeManagedMemory
    cudaDevAttrManagedMemory = hip.chip.hipDeviceAttributeManagedMemory
    hipDeviceAttributeManagedMemory = hip.chip.hipDeviceAttributeManagedMemory
    hipDeviceAttributeMaxBlocksPerMultiProcessor = hip.chip.hipDeviceAttributeMaxBlocksPerMultiProcessor
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = hip.chip.hipDeviceAttributeMaxBlockDimX
    cudaDevAttrMaxBlockDimX = hip.chip.hipDeviceAttributeMaxBlockDimX
    hipDeviceAttributeMaxBlockDimX = hip.chip.hipDeviceAttributeMaxBlockDimX
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = hip.chip.hipDeviceAttributeMaxBlockDimY
    cudaDevAttrMaxBlockDimY = hip.chip.hipDeviceAttributeMaxBlockDimY
    hipDeviceAttributeMaxBlockDimY = hip.chip.hipDeviceAttributeMaxBlockDimY
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = hip.chip.hipDeviceAttributeMaxBlockDimZ
    cudaDevAttrMaxBlockDimZ = hip.chip.hipDeviceAttributeMaxBlockDimZ
    hipDeviceAttributeMaxBlockDimZ = hip.chip.hipDeviceAttributeMaxBlockDimZ
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = hip.chip.hipDeviceAttributeMaxGridDimX
    cudaDevAttrMaxGridDimX = hip.chip.hipDeviceAttributeMaxGridDimX
    hipDeviceAttributeMaxGridDimX = hip.chip.hipDeviceAttributeMaxGridDimX
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = hip.chip.hipDeviceAttributeMaxGridDimY
    cudaDevAttrMaxGridDimY = hip.chip.hipDeviceAttributeMaxGridDimY
    hipDeviceAttributeMaxGridDimY = hip.chip.hipDeviceAttributeMaxGridDimY
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = hip.chip.hipDeviceAttributeMaxGridDimZ
    cudaDevAttrMaxGridDimZ = hip.chip.hipDeviceAttributeMaxGridDimZ
    hipDeviceAttributeMaxGridDimZ = hip.chip.hipDeviceAttributeMaxGridDimZ
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = hip.chip.hipDeviceAttributeMaxSurface1D
    cudaDevAttrMaxSurface1DWidth = hip.chip.hipDeviceAttributeMaxSurface1D
    hipDeviceAttributeMaxSurface1D = hip.chip.hipDeviceAttributeMaxSurface1D
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = hip.chip.hipDeviceAttributeMaxSurface1DLayered
    cudaDevAttrMaxSurface1DLayeredWidth = hip.chip.hipDeviceAttributeMaxSurface1DLayered
    hipDeviceAttributeMaxSurface1DLayered = hip.chip.hipDeviceAttributeMaxSurface1DLayered
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = hip.chip.hipDeviceAttributeMaxSurface2D
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = hip.chip.hipDeviceAttributeMaxSurface2D
    cudaDevAttrMaxSurface2DHeight = hip.chip.hipDeviceAttributeMaxSurface2D
    cudaDevAttrMaxSurface2DWidth = hip.chip.hipDeviceAttributeMaxSurface2D
    hipDeviceAttributeMaxSurface2D = hip.chip.hipDeviceAttributeMaxSurface2D
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = hip.chip.hipDeviceAttributeMaxSurface2DLayered
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = hip.chip.hipDeviceAttributeMaxSurface2DLayered
    cudaDevAttrMaxSurface2DLayeredHeight = hip.chip.hipDeviceAttributeMaxSurface2DLayered
    cudaDevAttrMaxSurface2DLayeredWidth = hip.chip.hipDeviceAttributeMaxSurface2DLayered
    hipDeviceAttributeMaxSurface2DLayered = hip.chip.hipDeviceAttributeMaxSurface2DLayered
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = hip.chip.hipDeviceAttributeMaxSurface3D
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = hip.chip.hipDeviceAttributeMaxSurface3D
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = hip.chip.hipDeviceAttributeMaxSurface3D
    cudaDevAttrMaxSurface3DDepth = hip.chip.hipDeviceAttributeMaxSurface3D
    cudaDevAttrMaxSurface3DHeight = hip.chip.hipDeviceAttributeMaxSurface3D
    cudaDevAttrMaxSurface3DWidth = hip.chip.hipDeviceAttributeMaxSurface3D
    hipDeviceAttributeMaxSurface3D = hip.chip.hipDeviceAttributeMaxSurface3D
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = hip.chip.hipDeviceAttributeMaxSurfaceCubemap
    cudaDevAttrMaxSurfaceCubemapWidth = hip.chip.hipDeviceAttributeMaxSurfaceCubemap
    hipDeviceAttributeMaxSurfaceCubemap = hip.chip.hipDeviceAttributeMaxSurfaceCubemap
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = hip.chip.hipDeviceAttributeMaxSurfaceCubemapLayered
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = hip.chip.hipDeviceAttributeMaxSurfaceCubemapLayered
    hipDeviceAttributeMaxSurfaceCubemapLayered = hip.chip.hipDeviceAttributeMaxSurfaceCubemapLayered
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = hip.chip.hipDeviceAttributeMaxTexture1DWidth
    cudaDevAttrMaxTexture1DWidth = hip.chip.hipDeviceAttributeMaxTexture1DWidth
    hipDeviceAttributeMaxTexture1DWidth = hip.chip.hipDeviceAttributeMaxTexture1DWidth
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = hip.chip.hipDeviceAttributeMaxTexture1DLayered
    cudaDevAttrMaxTexture1DLayeredWidth = hip.chip.hipDeviceAttributeMaxTexture1DLayered
    hipDeviceAttributeMaxTexture1DLayered = hip.chip.hipDeviceAttributeMaxTexture1DLayered
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = hip.chip.hipDeviceAttributeMaxTexture1DLinear
    cudaDevAttrMaxTexture1DLinearWidth = hip.chip.hipDeviceAttributeMaxTexture1DLinear
    hipDeviceAttributeMaxTexture1DLinear = hip.chip.hipDeviceAttributeMaxTexture1DLinear
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = hip.chip.hipDeviceAttributeMaxTexture1DMipmap
    cudaDevAttrMaxTexture1DMipmappedWidth = hip.chip.hipDeviceAttributeMaxTexture1DMipmap
    hipDeviceAttributeMaxTexture1DMipmap = hip.chip.hipDeviceAttributeMaxTexture1DMipmap
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = hip.chip.hipDeviceAttributeMaxTexture2DWidth
    cudaDevAttrMaxTexture2DWidth = hip.chip.hipDeviceAttributeMaxTexture2DWidth
    hipDeviceAttributeMaxTexture2DWidth = hip.chip.hipDeviceAttributeMaxTexture2DWidth
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = hip.chip.hipDeviceAttributeMaxTexture2DHeight
    cudaDevAttrMaxTexture2DHeight = hip.chip.hipDeviceAttributeMaxTexture2DHeight
    hipDeviceAttributeMaxTexture2DHeight = hip.chip.hipDeviceAttributeMaxTexture2DHeight
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = hip.chip.hipDeviceAttributeMaxTexture2DGather
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = hip.chip.hipDeviceAttributeMaxTexture2DGather
    cudaDevAttrMaxTexture2DGatherHeight = hip.chip.hipDeviceAttributeMaxTexture2DGather
    cudaDevAttrMaxTexture2DGatherWidth = hip.chip.hipDeviceAttributeMaxTexture2DGather
    hipDeviceAttributeMaxTexture2DGather = hip.chip.hipDeviceAttributeMaxTexture2DGather
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = hip.chip.hipDeviceAttributeMaxTexture2DLayered
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = hip.chip.hipDeviceAttributeMaxTexture2DLayered
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = hip.chip.hipDeviceAttributeMaxTexture2DLayered
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = hip.chip.hipDeviceAttributeMaxTexture2DLayered
    cudaDevAttrMaxTexture2DLayeredHeight = hip.chip.hipDeviceAttributeMaxTexture2DLayered
    cudaDevAttrMaxTexture2DLayeredWidth = hip.chip.hipDeviceAttributeMaxTexture2DLayered
    hipDeviceAttributeMaxTexture2DLayered = hip.chip.hipDeviceAttributeMaxTexture2DLayered
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = hip.chip.hipDeviceAttributeMaxTexture2DLinear
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = hip.chip.hipDeviceAttributeMaxTexture2DLinear
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = hip.chip.hipDeviceAttributeMaxTexture2DLinear
    cudaDevAttrMaxTexture2DLinearHeight = hip.chip.hipDeviceAttributeMaxTexture2DLinear
    cudaDevAttrMaxTexture2DLinearPitch = hip.chip.hipDeviceAttributeMaxTexture2DLinear
    cudaDevAttrMaxTexture2DLinearWidth = hip.chip.hipDeviceAttributeMaxTexture2DLinear
    hipDeviceAttributeMaxTexture2DLinear = hip.chip.hipDeviceAttributeMaxTexture2DLinear
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = hip.chip.hipDeviceAttributeMaxTexture2DMipmap
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = hip.chip.hipDeviceAttributeMaxTexture2DMipmap
    cudaDevAttrMaxTexture2DMipmappedHeight = hip.chip.hipDeviceAttributeMaxTexture2DMipmap
    cudaDevAttrMaxTexture2DMipmappedWidth = hip.chip.hipDeviceAttributeMaxTexture2DMipmap
    hipDeviceAttributeMaxTexture2DMipmap = hip.chip.hipDeviceAttributeMaxTexture2DMipmap
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = hip.chip.hipDeviceAttributeMaxTexture3DWidth
    cudaDevAttrMaxTexture3DWidth = hip.chip.hipDeviceAttributeMaxTexture3DWidth
    hipDeviceAttributeMaxTexture3DWidth = hip.chip.hipDeviceAttributeMaxTexture3DWidth
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = hip.chip.hipDeviceAttributeMaxTexture3DHeight
    cudaDevAttrMaxTexture3DHeight = hip.chip.hipDeviceAttributeMaxTexture3DHeight
    hipDeviceAttributeMaxTexture3DHeight = hip.chip.hipDeviceAttributeMaxTexture3DHeight
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = hip.chip.hipDeviceAttributeMaxTexture3DDepth
    cudaDevAttrMaxTexture3DDepth = hip.chip.hipDeviceAttributeMaxTexture3DDepth
    hipDeviceAttributeMaxTexture3DDepth = hip.chip.hipDeviceAttributeMaxTexture3DDepth
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = hip.chip.hipDeviceAttributeMaxTexture3DAlt
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = hip.chip.hipDeviceAttributeMaxTexture3DAlt
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = hip.chip.hipDeviceAttributeMaxTexture3DAlt
    cudaDevAttrMaxTexture3DDepthAlt = hip.chip.hipDeviceAttributeMaxTexture3DAlt
    cudaDevAttrMaxTexture3DHeightAlt = hip.chip.hipDeviceAttributeMaxTexture3DAlt
    cudaDevAttrMaxTexture3DWidthAlt = hip.chip.hipDeviceAttributeMaxTexture3DAlt
    hipDeviceAttributeMaxTexture3DAlt = hip.chip.hipDeviceAttributeMaxTexture3DAlt
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = hip.chip.hipDeviceAttributeMaxTextureCubemap
    cudaDevAttrMaxTextureCubemapWidth = hip.chip.hipDeviceAttributeMaxTextureCubemap
    hipDeviceAttributeMaxTextureCubemap = hip.chip.hipDeviceAttributeMaxTextureCubemap
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = hip.chip.hipDeviceAttributeMaxTextureCubemapLayered
    cudaDevAttrMaxTextureCubemapLayeredWidth = hip.chip.hipDeviceAttributeMaxTextureCubemapLayered
    hipDeviceAttributeMaxTextureCubemapLayered = hip.chip.hipDeviceAttributeMaxTextureCubemapLayered
    hipDeviceAttributeMaxThreadsDim = hip.chip.hipDeviceAttributeMaxThreadsDim
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = hip.chip.hipDeviceAttributeMaxThreadsPerBlock
    cudaDevAttrMaxThreadsPerBlock = hip.chip.hipDeviceAttributeMaxThreadsPerBlock
    hipDeviceAttributeMaxThreadsPerBlock = hip.chip.hipDeviceAttributeMaxThreadsPerBlock
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = hip.chip.hipDeviceAttributeMaxThreadsPerMultiProcessor
    cudaDevAttrMaxThreadsPerMultiProcessor = hip.chip.hipDeviceAttributeMaxThreadsPerMultiProcessor
    hipDeviceAttributeMaxThreadsPerMultiProcessor = hip.chip.hipDeviceAttributeMaxThreadsPerMultiProcessor
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = hip.chip.hipDeviceAttributeMaxPitch
    cudaDevAttrMaxPitch = hip.chip.hipDeviceAttributeMaxPitch
    hipDeviceAttributeMaxPitch = hip.chip.hipDeviceAttributeMaxPitch
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = hip.chip.hipDeviceAttributeMemoryBusWidth
    cudaDevAttrGlobalMemoryBusWidth = hip.chip.hipDeviceAttributeMemoryBusWidth
    hipDeviceAttributeMemoryBusWidth = hip.chip.hipDeviceAttributeMemoryBusWidth
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = hip.chip.hipDeviceAttributeMemoryClockRate
    cudaDevAttrMemoryClockRate = hip.chip.hipDeviceAttributeMemoryClockRate
    hipDeviceAttributeMemoryClockRate = hip.chip.hipDeviceAttributeMemoryClockRate
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = hip.chip.hipDeviceAttributeComputeCapabilityMinor
    cudaDevAttrComputeCapabilityMinor = hip.chip.hipDeviceAttributeComputeCapabilityMinor
    hipDeviceAttributeComputeCapabilityMinor = hip.chip.hipDeviceAttributeComputeCapabilityMinor
    cudaDevAttrMultiGpuBoardGroupID = hip.chip.hipDeviceAttributeMultiGpuBoardGroupID
    hipDeviceAttributeMultiGpuBoardGroupID = hip.chip.hipDeviceAttributeMultiGpuBoardGroupID
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = hip.chip.hipDeviceAttributeMultiprocessorCount
    cudaDevAttrMultiProcessorCount = hip.chip.hipDeviceAttributeMultiprocessorCount
    hipDeviceAttributeMultiprocessorCount = hip.chip.hipDeviceAttributeMultiprocessorCount
    hipDeviceAttributeName = hip.chip.hipDeviceAttributeName
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = hip.chip.hipDeviceAttributePageableMemoryAccess
    cudaDevAttrPageableMemoryAccess = hip.chip.hipDeviceAttributePageableMemoryAccess
    hipDeviceAttributePageableMemoryAccess = hip.chip.hipDeviceAttributePageableMemoryAccess
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = hip.chip.hipDeviceAttributePageableMemoryAccessUsesHostPageTables
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = hip.chip.hipDeviceAttributePageableMemoryAccessUsesHostPageTables
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables = hip.chip.hipDeviceAttributePageableMemoryAccessUsesHostPageTables
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = hip.chip.hipDeviceAttributePciBusId
    cudaDevAttrPciBusId = hip.chip.hipDeviceAttributePciBusId
    hipDeviceAttributePciBusId = hip.chip.hipDeviceAttributePciBusId
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = hip.chip.hipDeviceAttributePciDeviceId
    cudaDevAttrPciDeviceId = hip.chip.hipDeviceAttributePciDeviceId
    hipDeviceAttributePciDeviceId = hip.chip.hipDeviceAttributePciDeviceId
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = hip.chip.hipDeviceAttributePciDomainID
    cudaDevAttrPciDomainId = hip.chip.hipDeviceAttributePciDomainID
    hipDeviceAttributePciDomainID = hip.chip.hipDeviceAttributePciDomainID
    hipDeviceAttributePersistingL2CacheMaxSize = hip.chip.hipDeviceAttributePersistingL2CacheMaxSize
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = hip.chip.hipDeviceAttributeMaxRegistersPerBlock
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = hip.chip.hipDeviceAttributeMaxRegistersPerBlock
    cudaDevAttrMaxRegistersPerBlock = hip.chip.hipDeviceAttributeMaxRegistersPerBlock
    hipDeviceAttributeMaxRegistersPerBlock = hip.chip.hipDeviceAttributeMaxRegistersPerBlock
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = hip.chip.hipDeviceAttributeMaxRegistersPerMultiprocessor
    cudaDevAttrMaxRegistersPerMultiprocessor = hip.chip.hipDeviceAttributeMaxRegistersPerMultiprocessor
    hipDeviceAttributeMaxRegistersPerMultiprocessor = hip.chip.hipDeviceAttributeMaxRegistersPerMultiprocessor
    hipDeviceAttributeReservedSharedMemPerBlock = hip.chip.hipDeviceAttributeReservedSharedMemPerBlock
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = hip.chip.hipDeviceAttributeMaxSharedMemoryPerBlock
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = hip.chip.hipDeviceAttributeMaxSharedMemoryPerBlock
    cudaDevAttrMaxSharedMemoryPerBlock = hip.chip.hipDeviceAttributeMaxSharedMemoryPerBlock
    hipDeviceAttributeMaxSharedMemoryPerBlock = hip.chip.hipDeviceAttributeMaxSharedMemoryPerBlock
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = hip.chip.hipDeviceAttributeSharedMemPerBlockOptin
    cudaDevAttrMaxSharedMemoryPerBlockOptin = hip.chip.hipDeviceAttributeSharedMemPerBlockOptin
    hipDeviceAttributeSharedMemPerBlockOptin = hip.chip.hipDeviceAttributeSharedMemPerBlockOptin
    hipDeviceAttributeSharedMemPerMultiprocessor = hip.chip.hipDeviceAttributeSharedMemPerMultiprocessor
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = hip.chip.hipDeviceAttributeSingleToDoublePrecisionPerfRatio
    cudaDevAttrSingleToDoublePrecisionPerfRatio = hip.chip.hipDeviceAttributeSingleToDoublePrecisionPerfRatio
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio = hip.chip.hipDeviceAttributeSingleToDoublePrecisionPerfRatio
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = hip.chip.hipDeviceAttributeStreamPrioritiesSupported
    cudaDevAttrStreamPrioritiesSupported = hip.chip.hipDeviceAttributeStreamPrioritiesSupported
    hipDeviceAttributeStreamPrioritiesSupported = hip.chip.hipDeviceAttributeStreamPrioritiesSupported
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = hip.chip.hipDeviceAttributeSurfaceAlignment
    cudaDevAttrSurfaceAlignment = hip.chip.hipDeviceAttributeSurfaceAlignment
    hipDeviceAttributeSurfaceAlignment = hip.chip.hipDeviceAttributeSurfaceAlignment
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = hip.chip.hipDeviceAttributeTccDriver
    cudaDevAttrTccDriver = hip.chip.hipDeviceAttributeTccDriver
    hipDeviceAttributeTccDriver = hip.chip.hipDeviceAttributeTccDriver
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = hip.chip.hipDeviceAttributeTextureAlignment
    cudaDevAttrTextureAlignment = hip.chip.hipDeviceAttributeTextureAlignment
    hipDeviceAttributeTextureAlignment = hip.chip.hipDeviceAttributeTextureAlignment
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = hip.chip.hipDeviceAttributeTexturePitchAlignment
    cudaDevAttrTexturePitchAlignment = hip.chip.hipDeviceAttributeTexturePitchAlignment
    hipDeviceAttributeTexturePitchAlignment = hip.chip.hipDeviceAttributeTexturePitchAlignment
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = hip.chip.hipDeviceAttributeTotalConstantMemory
    cudaDevAttrTotalConstantMemory = hip.chip.hipDeviceAttributeTotalConstantMemory
    hipDeviceAttributeTotalConstantMemory = hip.chip.hipDeviceAttributeTotalConstantMemory
    hipDeviceAttributeTotalGlobalMem = hip.chip.hipDeviceAttributeTotalGlobalMem
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = hip.chip.hipDeviceAttributeUnifiedAddressing
    cudaDevAttrUnifiedAddressing = hip.chip.hipDeviceAttributeUnifiedAddressing
    hipDeviceAttributeUnifiedAddressing = hip.chip.hipDeviceAttributeUnifiedAddressing
    hipDeviceAttributeUuid = hip.chip.hipDeviceAttributeUuid
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = hip.chip.hipDeviceAttributeWarpSize
    cudaDevAttrWarpSize = hip.chip.hipDeviceAttributeWarpSize
    hipDeviceAttributeWarpSize = hip.chip.hipDeviceAttributeWarpSize
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = hip.chip.hipDeviceAttributeMemoryPoolsSupported
    cudaDevAttrMemoryPoolsSupported = hip.chip.hipDeviceAttributeMemoryPoolsSupported
    hipDeviceAttributeMemoryPoolsSupported = hip.chip.hipDeviceAttributeMemoryPoolsSupported
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = hip.chip.hipDeviceAttributeVirtualMemoryManagementSupported
    hipDeviceAttributeVirtualMemoryManagementSupported = hip.chip.hipDeviceAttributeVirtualMemoryManagementSupported
    hipDeviceAttributeCudaCompatibleEnd = hip.chip.hipDeviceAttributeCudaCompatibleEnd
    hipDeviceAttributeAmdSpecificBegin = hip.chip.hipDeviceAttributeAmdSpecificBegin
    hipDeviceAttributeClockInstructionRate = hip.chip.hipDeviceAttributeClockInstructionRate
    hipDeviceAttributeArch = hip.chip.hipDeviceAttributeArch
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = hip.chip.hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = hip.chip.hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = hip.chip.hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
    hipDeviceAttributeGcnArch = hip.chip.hipDeviceAttributeGcnArch
    hipDeviceAttributeGcnArchName = hip.chip.hipDeviceAttributeGcnArchName
    hipDeviceAttributeHdpMemFlushCntl = hip.chip.hipDeviceAttributeHdpMemFlushCntl
    hipDeviceAttributeHdpRegFlushCntl = hip.chip.hipDeviceAttributeHdpRegFlushCntl
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = hip.chip.hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = hip.chip.hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = hip.chip.hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = hip.chip.hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem
    hipDeviceAttributeIsLargeBar = hip.chip.hipDeviceAttributeIsLargeBar
    hipDeviceAttributeAsicRevision = hip.chip.hipDeviceAttributeAsicRevision
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = hip.chip.hipDeviceAttributeCanUseStreamWaitValue
    cudaDevAttrReserved94 = hip.chip.hipDeviceAttributeCanUseStreamWaitValue
    hipDeviceAttributeCanUseStreamWaitValue = hip.chip.hipDeviceAttributeCanUseStreamWaitValue
    hipDeviceAttributeImageSupport = hip.chip.hipDeviceAttributeImageSupport
    hipDeviceAttributePhysicalMultiProcessorCount = hip.chip.hipDeviceAttributePhysicalMultiProcessorCount
    hipDeviceAttributeFineGrainSupport = hip.chip.hipDeviceAttributeFineGrainSupport
    hipDeviceAttributeWallClockRate = hip.chip.hipDeviceAttributeWallClockRate
    hipDeviceAttributeAmdSpecificEnd = hip.chip.hipDeviceAttributeAmdSpecificEnd
    hipDeviceAttributeVendorSpecificBegin = hip.chip.hipDeviceAttributeVendorSpecificBegin
ctypedef CUdevice_attribute CUdevice_attribute_enum
ctypedef CUdevice_attribute cudaDeviceAttr
cdef enum CUcomputemode:
    CU_COMPUTEMODE_DEFAULT = hip.chip.hipComputeModeDefault
    cudaComputeModeDefault = hip.chip.hipComputeModeDefault
    hipComputeModeDefault = hip.chip.hipComputeModeDefault
    CU_COMPUTEMODE_EXCLUSIVE = hip.chip.hipComputeModeExclusive
    cudaComputeModeExclusive = hip.chip.hipComputeModeExclusive
    hipComputeModeExclusive = hip.chip.hipComputeModeExclusive
    CU_COMPUTEMODE_PROHIBITED = hip.chip.hipComputeModeProhibited
    cudaComputeModeProhibited = hip.chip.hipComputeModeProhibited
    hipComputeModeProhibited = hip.chip.hipComputeModeProhibited
    CU_COMPUTEMODE_EXCLUSIVE_PROCESS = hip.chip.hipComputeModeExclusiveProcess
    cudaComputeModeExclusiveProcess = hip.chip.hipComputeModeExclusiveProcess
    hipComputeModeExclusiveProcess = hip.chip.hipComputeModeExclusiveProcess
ctypedef CUcomputemode CUcomputemode_enum
ctypedef CUcomputemode cudaComputeMode
from hip.chip cimport hipDeviceptr_t as CUdeviceptr
from hip.chip cimport hipDeviceptr_t as CUdeviceptr_v1
from hip.chip cimport hipDeviceptr_t as CUdeviceptr_v2
cdef enum cudaChannelFormatKind:
    cudaChannelFormatKindSigned = hip.chip.hipChannelFormatKindSigned
    hipChannelFormatKindSigned = hip.chip.hipChannelFormatKindSigned
    cudaChannelFormatKindUnsigned = hip.chip.hipChannelFormatKindUnsigned
    hipChannelFormatKindUnsigned = hip.chip.hipChannelFormatKindUnsigned
    cudaChannelFormatKindFloat = hip.chip.hipChannelFormatKindFloat
    hipChannelFormatKindFloat = hip.chip.hipChannelFormatKindFloat
    cudaChannelFormatKindNone = hip.chip.hipChannelFormatKindNone
    hipChannelFormatKindNone = hip.chip.hipChannelFormatKindNone
from hip.chip cimport hipChannelFormatDesc as cudaChannelFormatDesc
cdef enum CUarray_format:
    CU_AD_FORMAT_UNSIGNED_INT8 = hip.chip.HIP_AD_FORMAT_UNSIGNED_INT8
    HIP_AD_FORMAT_UNSIGNED_INT8 = hip.chip.HIP_AD_FORMAT_UNSIGNED_INT8
    CU_AD_FORMAT_UNSIGNED_INT16 = hip.chip.HIP_AD_FORMAT_UNSIGNED_INT16
    HIP_AD_FORMAT_UNSIGNED_INT16 = hip.chip.HIP_AD_FORMAT_UNSIGNED_INT16
    CU_AD_FORMAT_UNSIGNED_INT32 = hip.chip.HIP_AD_FORMAT_UNSIGNED_INT32
    HIP_AD_FORMAT_UNSIGNED_INT32 = hip.chip.HIP_AD_FORMAT_UNSIGNED_INT32
    CU_AD_FORMAT_SIGNED_INT8 = hip.chip.HIP_AD_FORMAT_SIGNED_INT8
    HIP_AD_FORMAT_SIGNED_INT8 = hip.chip.HIP_AD_FORMAT_SIGNED_INT8
    CU_AD_FORMAT_SIGNED_INT16 = hip.chip.HIP_AD_FORMAT_SIGNED_INT16
    HIP_AD_FORMAT_SIGNED_INT16 = hip.chip.HIP_AD_FORMAT_SIGNED_INT16
    CU_AD_FORMAT_SIGNED_INT32 = hip.chip.HIP_AD_FORMAT_SIGNED_INT32
    HIP_AD_FORMAT_SIGNED_INT32 = hip.chip.HIP_AD_FORMAT_SIGNED_INT32
    CU_AD_FORMAT_HALF = hip.chip.HIP_AD_FORMAT_HALF
    HIP_AD_FORMAT_HALF = hip.chip.HIP_AD_FORMAT_HALF
    CU_AD_FORMAT_FLOAT = hip.chip.HIP_AD_FORMAT_FLOAT
    HIP_AD_FORMAT_FLOAT = hip.chip.HIP_AD_FORMAT_FLOAT
ctypedef CUarray_format CUarray_format_enum
from hip.chip cimport HIP_ARRAY_DESCRIPTOR as CUDA_ARRAY_DESCRIPTOR
from hip.chip cimport HIP_ARRAY_DESCRIPTOR as CUDA_ARRAY_DESCRIPTOR_st
from hip.chip cimport HIP_ARRAY_DESCRIPTOR as CUDA_ARRAY_DESCRIPTOR_v1
from hip.chip cimport HIP_ARRAY_DESCRIPTOR as CUDA_ARRAY_DESCRIPTOR_v1_st
from hip.chip cimport HIP_ARRAY_DESCRIPTOR as CUDA_ARRAY_DESCRIPTOR_v2
from hip.chip cimport HIP_ARRAY3D_DESCRIPTOR as CUDA_ARRAY3D_DESCRIPTOR
from hip.chip cimport HIP_ARRAY3D_DESCRIPTOR as CUDA_ARRAY3D_DESCRIPTOR_st
from hip.chip cimport HIP_ARRAY3D_DESCRIPTOR as CUDA_ARRAY3D_DESCRIPTOR_v2
from hip.chip cimport hipArray as CUarray_st
from hip.chip cimport hipArray as cudaArray
from hip.chip cimport hip_Memcpy2D as CUDA_MEMCPY2D
from hip.chip cimport hip_Memcpy2D as CUDA_MEMCPY2D_st
from hip.chip cimport hip_Memcpy2D as CUDA_MEMCPY2D_v1
from hip.chip cimport hip_Memcpy2D as CUDA_MEMCPY2D_v1_st
from hip.chip cimport hip_Memcpy2D as CUDA_MEMCPY2D_v2
from hip.chip cimport hipMipmappedArray as CUmipmappedArray_st
from hip.chip cimport hipMipmappedArray as cudaMipmappedArray
cdef enum cudaResourceType:
    cudaResourceTypeArray = hip.chip.hipResourceTypeArray
    hipResourceTypeArray = hip.chip.hipResourceTypeArray
    cudaResourceTypeMipmappedArray = hip.chip.hipResourceTypeMipmappedArray
    hipResourceTypeMipmappedArray = hip.chip.hipResourceTypeMipmappedArray
    cudaResourceTypeLinear = hip.chip.hipResourceTypeLinear
    hipResourceTypeLinear = hip.chip.hipResourceTypeLinear
    cudaResourceTypePitch2D = hip.chip.hipResourceTypePitch2D
    hipResourceTypePitch2D = hip.chip.hipResourceTypePitch2D
cdef enum CUresourcetype_enum:
    CU_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    HIP_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    CU_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    HIP_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    CU_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D
    HIP_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D
ctypedef CUresourcetype_enum CUresourcetype
cdef enum CUaddress_mode_enum:
    CU_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    HIP_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    CU_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    HIP_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    CU_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    HIP_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    CU_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER
    HIP_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER
ctypedef CUaddress_mode_enum CUaddress_mode
cdef enum CUfilter_mode_enum:
    CU_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    HIP_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    CU_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR
    HIP_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR
ctypedef CUfilter_mode_enum CUfilter_mode
from hip.chip cimport HIP_TEXTURE_DESC_st as CUDA_TEXTURE_DESC_st
cdef enum cudaResourceViewFormat:
    cudaResViewFormatNone = hip.chip.hipResViewFormatNone
    hipResViewFormatNone = hip.chip.hipResViewFormatNone
    cudaResViewFormatUnsignedChar1 = hip.chip.hipResViewFormatUnsignedChar1
    hipResViewFormatUnsignedChar1 = hip.chip.hipResViewFormatUnsignedChar1
    cudaResViewFormatUnsignedChar2 = hip.chip.hipResViewFormatUnsignedChar2
    hipResViewFormatUnsignedChar2 = hip.chip.hipResViewFormatUnsignedChar2
    cudaResViewFormatUnsignedChar4 = hip.chip.hipResViewFormatUnsignedChar4
    hipResViewFormatUnsignedChar4 = hip.chip.hipResViewFormatUnsignedChar4
    cudaResViewFormatSignedChar1 = hip.chip.hipResViewFormatSignedChar1
    hipResViewFormatSignedChar1 = hip.chip.hipResViewFormatSignedChar1
    cudaResViewFormatSignedChar2 = hip.chip.hipResViewFormatSignedChar2
    hipResViewFormatSignedChar2 = hip.chip.hipResViewFormatSignedChar2
    cudaResViewFormatSignedChar4 = hip.chip.hipResViewFormatSignedChar4
    hipResViewFormatSignedChar4 = hip.chip.hipResViewFormatSignedChar4
    cudaResViewFormatUnsignedShort1 = hip.chip.hipResViewFormatUnsignedShort1
    hipResViewFormatUnsignedShort1 = hip.chip.hipResViewFormatUnsignedShort1
    cudaResViewFormatUnsignedShort2 = hip.chip.hipResViewFormatUnsignedShort2
    hipResViewFormatUnsignedShort2 = hip.chip.hipResViewFormatUnsignedShort2
    cudaResViewFormatUnsignedShort4 = hip.chip.hipResViewFormatUnsignedShort4
    hipResViewFormatUnsignedShort4 = hip.chip.hipResViewFormatUnsignedShort4
    cudaResViewFormatSignedShort1 = hip.chip.hipResViewFormatSignedShort1
    hipResViewFormatSignedShort1 = hip.chip.hipResViewFormatSignedShort1
    cudaResViewFormatSignedShort2 = hip.chip.hipResViewFormatSignedShort2
    hipResViewFormatSignedShort2 = hip.chip.hipResViewFormatSignedShort2
    cudaResViewFormatSignedShort4 = hip.chip.hipResViewFormatSignedShort4
    hipResViewFormatSignedShort4 = hip.chip.hipResViewFormatSignedShort4
    cudaResViewFormatUnsignedInt1 = hip.chip.hipResViewFormatUnsignedInt1
    hipResViewFormatUnsignedInt1 = hip.chip.hipResViewFormatUnsignedInt1
    cudaResViewFormatUnsignedInt2 = hip.chip.hipResViewFormatUnsignedInt2
    hipResViewFormatUnsignedInt2 = hip.chip.hipResViewFormatUnsignedInt2
    cudaResViewFormatUnsignedInt4 = hip.chip.hipResViewFormatUnsignedInt4
    hipResViewFormatUnsignedInt4 = hip.chip.hipResViewFormatUnsignedInt4
    cudaResViewFormatSignedInt1 = hip.chip.hipResViewFormatSignedInt1
    hipResViewFormatSignedInt1 = hip.chip.hipResViewFormatSignedInt1
    cudaResViewFormatSignedInt2 = hip.chip.hipResViewFormatSignedInt2
    hipResViewFormatSignedInt2 = hip.chip.hipResViewFormatSignedInt2
    cudaResViewFormatSignedInt4 = hip.chip.hipResViewFormatSignedInt4
    hipResViewFormatSignedInt4 = hip.chip.hipResViewFormatSignedInt4
    cudaResViewFormatHalf1 = hip.chip.hipResViewFormatHalf1
    hipResViewFormatHalf1 = hip.chip.hipResViewFormatHalf1
    cudaResViewFormatHalf2 = hip.chip.hipResViewFormatHalf2
    hipResViewFormatHalf2 = hip.chip.hipResViewFormatHalf2
    cudaResViewFormatHalf4 = hip.chip.hipResViewFormatHalf4
    hipResViewFormatHalf4 = hip.chip.hipResViewFormatHalf4
    cudaResViewFormatFloat1 = hip.chip.hipResViewFormatFloat1
    hipResViewFormatFloat1 = hip.chip.hipResViewFormatFloat1
    cudaResViewFormatFloat2 = hip.chip.hipResViewFormatFloat2
    hipResViewFormatFloat2 = hip.chip.hipResViewFormatFloat2
    cudaResViewFormatFloat4 = hip.chip.hipResViewFormatFloat4
    hipResViewFormatFloat4 = hip.chip.hipResViewFormatFloat4
    cudaResViewFormatUnsignedBlockCompressed1 = hip.chip.hipResViewFormatUnsignedBlockCompressed1
    hipResViewFormatUnsignedBlockCompressed1 = hip.chip.hipResViewFormatUnsignedBlockCompressed1
    cudaResViewFormatUnsignedBlockCompressed2 = hip.chip.hipResViewFormatUnsignedBlockCompressed2
    hipResViewFormatUnsignedBlockCompressed2 = hip.chip.hipResViewFormatUnsignedBlockCompressed2
    cudaResViewFormatUnsignedBlockCompressed3 = hip.chip.hipResViewFormatUnsignedBlockCompressed3
    hipResViewFormatUnsignedBlockCompressed3 = hip.chip.hipResViewFormatUnsignedBlockCompressed3
    cudaResViewFormatUnsignedBlockCompressed4 = hip.chip.hipResViewFormatUnsignedBlockCompressed4
    hipResViewFormatUnsignedBlockCompressed4 = hip.chip.hipResViewFormatUnsignedBlockCompressed4
    cudaResViewFormatSignedBlockCompressed4 = hip.chip.hipResViewFormatSignedBlockCompressed4
    hipResViewFormatSignedBlockCompressed4 = hip.chip.hipResViewFormatSignedBlockCompressed4
    cudaResViewFormatUnsignedBlockCompressed5 = hip.chip.hipResViewFormatUnsignedBlockCompressed5
    hipResViewFormatUnsignedBlockCompressed5 = hip.chip.hipResViewFormatUnsignedBlockCompressed5
    cudaResViewFormatSignedBlockCompressed5 = hip.chip.hipResViewFormatSignedBlockCompressed5
    hipResViewFormatSignedBlockCompressed5 = hip.chip.hipResViewFormatSignedBlockCompressed5
    cudaResViewFormatUnsignedBlockCompressed6H = hip.chip.hipResViewFormatUnsignedBlockCompressed6H
    hipResViewFormatUnsignedBlockCompressed6H = hip.chip.hipResViewFormatUnsignedBlockCompressed6H
    cudaResViewFormatSignedBlockCompressed6H = hip.chip.hipResViewFormatSignedBlockCompressed6H
    hipResViewFormatSignedBlockCompressed6H = hip.chip.hipResViewFormatSignedBlockCompressed6H
    cudaResViewFormatUnsignedBlockCompressed7 = hip.chip.hipResViewFormatUnsignedBlockCompressed7
    hipResViewFormatUnsignedBlockCompressed7 = hip.chip.hipResViewFormatUnsignedBlockCompressed7
cdef enum CUresourceViewFormat_enum:
    CU_RES_VIEW_FORMAT_NONE = hip.chip.HIP_RES_VIEW_FORMAT_NONE
    HIP_RES_VIEW_FORMAT_NONE = hip.chip.HIP_RES_VIEW_FORMAT_NONE
    CU_RES_VIEW_FORMAT_UINT_1X8 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_1X8
    HIP_RES_VIEW_FORMAT_UINT_1X8 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_1X8
    CU_RES_VIEW_FORMAT_UINT_2X8 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_2X8
    HIP_RES_VIEW_FORMAT_UINT_2X8 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_2X8
    CU_RES_VIEW_FORMAT_UINT_4X8 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_4X8
    HIP_RES_VIEW_FORMAT_UINT_4X8 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_4X8
    CU_RES_VIEW_FORMAT_SINT_1X8 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_1X8
    HIP_RES_VIEW_FORMAT_SINT_1X8 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_1X8
    CU_RES_VIEW_FORMAT_SINT_2X8 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_2X8
    HIP_RES_VIEW_FORMAT_SINT_2X8 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_2X8
    CU_RES_VIEW_FORMAT_SINT_4X8 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_4X8
    HIP_RES_VIEW_FORMAT_SINT_4X8 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_4X8
    CU_RES_VIEW_FORMAT_UINT_1X16 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_1X16
    HIP_RES_VIEW_FORMAT_UINT_1X16 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_1X16
    CU_RES_VIEW_FORMAT_UINT_2X16 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_2X16
    HIP_RES_VIEW_FORMAT_UINT_2X16 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_2X16
    CU_RES_VIEW_FORMAT_UINT_4X16 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_4X16
    HIP_RES_VIEW_FORMAT_UINT_4X16 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_4X16
    CU_RES_VIEW_FORMAT_SINT_1X16 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_1X16
    HIP_RES_VIEW_FORMAT_SINT_1X16 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_1X16
    CU_RES_VIEW_FORMAT_SINT_2X16 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_2X16
    HIP_RES_VIEW_FORMAT_SINT_2X16 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_2X16
    CU_RES_VIEW_FORMAT_SINT_4X16 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_4X16
    HIP_RES_VIEW_FORMAT_SINT_4X16 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_4X16
    CU_RES_VIEW_FORMAT_UINT_1X32 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_1X32
    HIP_RES_VIEW_FORMAT_UINT_1X32 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_1X32
    CU_RES_VIEW_FORMAT_UINT_2X32 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_2X32
    HIP_RES_VIEW_FORMAT_UINT_2X32 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_2X32
    CU_RES_VIEW_FORMAT_UINT_4X32 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_4X32
    HIP_RES_VIEW_FORMAT_UINT_4X32 = hip.chip.HIP_RES_VIEW_FORMAT_UINT_4X32
    CU_RES_VIEW_FORMAT_SINT_1X32 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_1X32
    HIP_RES_VIEW_FORMAT_SINT_1X32 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_1X32
    CU_RES_VIEW_FORMAT_SINT_2X32 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_2X32
    HIP_RES_VIEW_FORMAT_SINT_2X32 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_2X32
    CU_RES_VIEW_FORMAT_SINT_4X32 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_4X32
    HIP_RES_VIEW_FORMAT_SINT_4X32 = hip.chip.HIP_RES_VIEW_FORMAT_SINT_4X32
    CU_RES_VIEW_FORMAT_FLOAT_1X16 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_1X16
    HIP_RES_VIEW_FORMAT_FLOAT_1X16 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_1X16
    CU_RES_VIEW_FORMAT_FLOAT_2X16 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_2X16
    HIP_RES_VIEW_FORMAT_FLOAT_2X16 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_2X16
    CU_RES_VIEW_FORMAT_FLOAT_4X16 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_4X16
    HIP_RES_VIEW_FORMAT_FLOAT_4X16 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_4X16
    CU_RES_VIEW_FORMAT_FLOAT_1X32 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_1X32
    HIP_RES_VIEW_FORMAT_FLOAT_1X32 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_1X32
    CU_RES_VIEW_FORMAT_FLOAT_2X32 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_2X32
    HIP_RES_VIEW_FORMAT_FLOAT_2X32 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_2X32
    CU_RES_VIEW_FORMAT_FLOAT_4X32 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_4X32
    HIP_RES_VIEW_FORMAT_FLOAT_4X32 = hip.chip.HIP_RES_VIEW_FORMAT_FLOAT_4X32
    CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
    CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
    CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
    CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
    CU_RES_VIEW_FORMAT_SIGNED_BC4 = hip.chip.HIP_RES_VIEW_FORMAT_SIGNED_BC4
    HIP_RES_VIEW_FORMAT_SIGNED_BC4 = hip.chip.HIP_RES_VIEW_FORMAT_SIGNED_BC4
    CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
    CU_RES_VIEW_FORMAT_SIGNED_BC5 = hip.chip.HIP_RES_VIEW_FORMAT_SIGNED_BC5
    HIP_RES_VIEW_FORMAT_SIGNED_BC5 = hip.chip.HIP_RES_VIEW_FORMAT_SIGNED_BC5
    CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
    CU_RES_VIEW_FORMAT_SIGNED_BC6H = hip.chip.HIP_RES_VIEW_FORMAT_SIGNED_BC6H
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H = hip.chip.HIP_RES_VIEW_FORMAT_SIGNED_BC6H
    CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC7
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = hip.chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC7
ctypedef CUresourceViewFormat_enum CUresourceViewFormat
from hip.chip cimport hipResourceDesc as cudaResourceDesc
from hip.chip cimport HIP_RESOURCE_DESC_st as CUDA_RESOURCE_DESC_st
from hip.chip cimport hipResourceViewDesc as cudaResourceViewDesc
from hip.chip cimport HIP_RESOURCE_VIEW_DESC_st as CUDA_RESOURCE_VIEW_DESC_st
cdef enum cudaMemcpyKind:
    cudaMemcpyHostToHost = hip.chip.hipMemcpyHostToHost
    hipMemcpyHostToHost = hip.chip.hipMemcpyHostToHost
    cudaMemcpyHostToDevice = hip.chip.hipMemcpyHostToDevice
    hipMemcpyHostToDevice = hip.chip.hipMemcpyHostToDevice
    cudaMemcpyDeviceToHost = hip.chip.hipMemcpyDeviceToHost
    hipMemcpyDeviceToHost = hip.chip.hipMemcpyDeviceToHost
    cudaMemcpyDeviceToDevice = hip.chip.hipMemcpyDeviceToDevice
    hipMemcpyDeviceToDevice = hip.chip.hipMemcpyDeviceToDevice
    cudaMemcpyDefault = hip.chip.hipMemcpyDefault
    hipMemcpyDefault = hip.chip.hipMemcpyDefault
from hip.chip cimport hipPitchedPtr as cudaPitchedPtr
from hip.chip cimport hipExtent as cudaExtent
from hip.chip cimport hipPos as cudaPos
from hip.chip cimport hipMemcpy3DParms as cudaMemcpy3DParms
from hip.chip cimport HIP_MEMCPY3D as CUDA_MEMCPY3D
from hip.chip cimport HIP_MEMCPY3D as CUDA_MEMCPY3D_st
from hip.chip cimport HIP_MEMCPY3D as CUDA_MEMCPY3D_v1
from hip.chip cimport HIP_MEMCPY3D as CUDA_MEMCPY3D_v1_st
from hip.chip cimport HIP_MEMCPY3D as CUDA_MEMCPY3D_v2
cdef enum CUfunction_attribute:
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = hip.chip.HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = hip.chip.HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = hip.chip.HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = hip.chip.HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = hip.chip.HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = hip.chip.HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = hip.chip.HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = hip.chip.HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
    CU_FUNC_ATTRIBUTE_NUM_REGS = hip.chip.HIP_FUNC_ATTRIBUTE_NUM_REGS
    HIP_FUNC_ATTRIBUTE_NUM_REGS = hip.chip.HIP_FUNC_ATTRIBUTE_NUM_REGS
    CU_FUNC_ATTRIBUTE_PTX_VERSION = hip.chip.HIP_FUNC_ATTRIBUTE_PTX_VERSION
    HIP_FUNC_ATTRIBUTE_PTX_VERSION = hip.chip.HIP_FUNC_ATTRIBUTE_PTX_VERSION
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = hip.chip.HIP_FUNC_ATTRIBUTE_BINARY_VERSION
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION = hip.chip.HIP_FUNC_ATTRIBUTE_BINARY_VERSION
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = hip.chip.HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
    HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = hip.chip.HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = hip.chip.HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = hip.chip.HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = hip.chip.HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = hip.chip.HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    CU_FUNC_ATTRIBUTE_MAX = hip.chip.HIP_FUNC_ATTRIBUTE_MAX
    HIP_FUNC_ATTRIBUTE_MAX = hip.chip.HIP_FUNC_ATTRIBUTE_MAX
ctypedef CUfunction_attribute CUfunction_attribute_enum
cdef enum CUpointer_attribute:
    CU_POINTER_ATTRIBUTE_CONTEXT = hip.chip.HIP_POINTER_ATTRIBUTE_CONTEXT
    HIP_POINTER_ATTRIBUTE_CONTEXT = hip.chip.HIP_POINTER_ATTRIBUTE_CONTEXT
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = hip.chip.HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = hip.chip.HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = hip.chip.HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = hip.chip.HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
    CU_POINTER_ATTRIBUTE_HOST_POINTER = hip.chip.HIP_POINTER_ATTRIBUTE_HOST_POINTER
    HIP_POINTER_ATTRIBUTE_HOST_POINTER = hip.chip.HIP_POINTER_ATTRIBUTE_HOST_POINTER
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = hip.chip.HIP_POINTER_ATTRIBUTE_P2P_TOKENS
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS = hip.chip.HIP_POINTER_ATTRIBUTE_P2P_TOKENS
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = hip.chip.HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = hip.chip.HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
    CU_POINTER_ATTRIBUTE_BUFFER_ID = hip.chip.HIP_POINTER_ATTRIBUTE_BUFFER_ID
    HIP_POINTER_ATTRIBUTE_BUFFER_ID = hip.chip.HIP_POINTER_ATTRIBUTE_BUFFER_ID
    CU_POINTER_ATTRIBUTE_IS_MANAGED = hip.chip.HIP_POINTER_ATTRIBUTE_IS_MANAGED
    HIP_POINTER_ATTRIBUTE_IS_MANAGED = hip.chip.HIP_POINTER_ATTRIBUTE_IS_MANAGED
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = hip.chip.HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = hip.chip.HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = hip.chip.HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = hip.chip.HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = hip.chip.HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = hip.chip.HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = hip.chip.HIP_POINTER_ATTRIBUTE_RANGE_SIZE
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE = hip.chip.HIP_POINTER_ATTRIBUTE_RANGE_SIZE
    CU_POINTER_ATTRIBUTE_MAPPED = hip.chip.HIP_POINTER_ATTRIBUTE_MAPPED
    HIP_POINTER_ATTRIBUTE_MAPPED = hip.chip.HIP_POINTER_ATTRIBUTE_MAPPED
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = hip.chip.HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = hip.chip.HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = hip.chip.HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = hip.chip.HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = hip.chip.HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = hip.chip.HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = hip.chip.HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = hip.chip.HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
ctypedef CUpointer_attribute CUpointer_attribute_enum
from hip.chip cimport hipCreateChannelDesc as cudaCreateChannelDesc
cdef enum cudaTextureAddressMode:
    cudaAddressModeWrap = hip.chip.hipAddressModeWrap
    hipAddressModeWrap = hip.chip.hipAddressModeWrap
    cudaAddressModeClamp = hip.chip.hipAddressModeClamp
    hipAddressModeClamp = hip.chip.hipAddressModeClamp
    cudaAddressModeMirror = hip.chip.hipAddressModeMirror
    hipAddressModeMirror = hip.chip.hipAddressModeMirror
    cudaAddressModeBorder = hip.chip.hipAddressModeBorder
    hipAddressModeBorder = hip.chip.hipAddressModeBorder
cdef enum cudaTextureFilterMode:
    cudaFilterModePoint = hip.chip.hipFilterModePoint
    hipFilterModePoint = hip.chip.hipFilterModePoint
    cudaFilterModeLinear = hip.chip.hipFilterModeLinear
    hipFilterModeLinear = hip.chip.hipFilterModeLinear
cdef enum cudaTextureReadMode:
    cudaReadModeElementType = hip.chip.hipReadModeElementType
    hipReadModeElementType = hip.chip.hipReadModeElementType
    cudaReadModeNormalizedFloat = hip.chip.hipReadModeNormalizedFloat
    hipReadModeNormalizedFloat = hip.chip.hipReadModeNormalizedFloat
from hip.chip cimport textureReference as CUtexref_st
from hip.chip cimport textureReference as textureReference
from hip.chip cimport hipTextureDesc as cudaTextureDesc
from hip.chip cimport surfaceReference as surfaceReference
cdef enum cudaSurfaceBoundaryMode:
    cudaBoundaryModeZero = hip.chip.hipBoundaryModeZero
    hipBoundaryModeZero = hip.chip.hipBoundaryModeZero
    cudaBoundaryModeTrap = hip.chip.hipBoundaryModeTrap
    hipBoundaryModeTrap = hip.chip.hipBoundaryModeTrap
    cudaBoundaryModeClamp = hip.chip.hipBoundaryModeClamp
    hipBoundaryModeClamp = hip.chip.hipBoundaryModeClamp
from hip.chip cimport ihipCtx_t as CUctx_st
from hip.chip cimport hipDevice_t as CUdevice
from hip.chip cimport hipDevice_t as CUdevice_v1
cdef enum CUdevice_P2PAttribute:
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = hip.chip.hipDevP2PAttrPerformanceRank
    cudaDevP2PAttrPerformanceRank = hip.chip.hipDevP2PAttrPerformanceRank
    hipDevP2PAttrPerformanceRank = hip.chip.hipDevP2PAttrPerformanceRank
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = hip.chip.hipDevP2PAttrAccessSupported
    cudaDevP2PAttrAccessSupported = hip.chip.hipDevP2PAttrAccessSupported
    hipDevP2PAttrAccessSupported = hip.chip.hipDevP2PAttrAccessSupported
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = hip.chip.hipDevP2PAttrNativeAtomicSupported
    cudaDevP2PAttrNativeAtomicSupported = hip.chip.hipDevP2PAttrNativeAtomicSupported
    hipDevP2PAttrNativeAtomicSupported = hip.chip.hipDevP2PAttrNativeAtomicSupported
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = hip.chip.hipDevP2PAttrHipArrayAccessSupported
    CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED = hip.chip.hipDevP2PAttrHipArrayAccessSupported
    CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = hip.chip.hipDevP2PAttrHipArrayAccessSupported
    cudaDevP2PAttrCudaArrayAccessSupported = hip.chip.hipDevP2PAttrHipArrayAccessSupported
    hipDevP2PAttrHipArrayAccessSupported = hip.chip.hipDevP2PAttrHipArrayAccessSupported
ctypedef CUdevice_P2PAttribute CUdevice_P2PAttribute_enum
ctypedef CUdevice_P2PAttribute cudaDeviceP2PAttr
from hip.chip cimport ihipStream_t as CUstream_st
from hip.chip cimport hipIpcMemHandle_st as CUipcMemHandle_st
from hip.chip cimport hipIpcMemHandle_st as cudaIpcMemHandle_st
from hip.chip cimport hipIpcEventHandle_st as CUipcEventHandle_st
from hip.chip cimport hipIpcEventHandle_st as cudaIpcEventHandle_st
from hip.chip cimport ihipModule_t as CUmod_st
from hip.chip cimport ihipModuleSymbol_t as CUfunc_st
from hip.chip cimport ihipMemPoolHandle_t as CUmemPoolHandle_st
from hip.chip cimport hipFuncAttributes as cudaFuncAttributes
from hip.chip cimport ihipEvent_t as CUevent_st
cdef enum CUlimit:
    CU_LIMIT_STACK_SIZE = hip.chip.hipLimitStackSize
    cudaLimitStackSize = hip.chip.hipLimitStackSize
    hipLimitStackSize = hip.chip.hipLimitStackSize
    CU_LIMIT_PRINTF_FIFO_SIZE = hip.chip.hipLimitPrintfFifoSize
    cudaLimitPrintfFifoSize = hip.chip.hipLimitPrintfFifoSize
    hipLimitPrintfFifoSize = hip.chip.hipLimitPrintfFifoSize
    CU_LIMIT_MALLOC_HEAP_SIZE = hip.chip.hipLimitMallocHeapSize
    cudaLimitMallocHeapSize = hip.chip.hipLimitMallocHeapSize
    hipLimitMallocHeapSize = hip.chip.hipLimitMallocHeapSize
    hipLimitRange = hip.chip.hipLimitRange
ctypedef CUlimit CUlimit_enum
ctypedef CUlimit cudaLimit
cdef enum CUmem_advise:
    CU_MEM_ADVISE_SET_READ_MOSTLY = hip.chip.hipMemAdviseSetReadMostly
    cudaMemAdviseSetReadMostly = hip.chip.hipMemAdviseSetReadMostly
    hipMemAdviseSetReadMostly = hip.chip.hipMemAdviseSetReadMostly
    CU_MEM_ADVISE_UNSET_READ_MOSTLY = hip.chip.hipMemAdviseUnsetReadMostly
    cudaMemAdviseUnsetReadMostly = hip.chip.hipMemAdviseUnsetReadMostly
    hipMemAdviseUnsetReadMostly = hip.chip.hipMemAdviseUnsetReadMostly
    CU_MEM_ADVISE_SET_PREFERRED_LOCATION = hip.chip.hipMemAdviseSetPreferredLocation
    cudaMemAdviseSetPreferredLocation = hip.chip.hipMemAdviseSetPreferredLocation
    hipMemAdviseSetPreferredLocation = hip.chip.hipMemAdviseSetPreferredLocation
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = hip.chip.hipMemAdviseUnsetPreferredLocation
    cudaMemAdviseUnsetPreferredLocation = hip.chip.hipMemAdviseUnsetPreferredLocation
    hipMemAdviseUnsetPreferredLocation = hip.chip.hipMemAdviseUnsetPreferredLocation
    CU_MEM_ADVISE_SET_ACCESSED_BY = hip.chip.hipMemAdviseSetAccessedBy
    cudaMemAdviseSetAccessedBy = hip.chip.hipMemAdviseSetAccessedBy
    hipMemAdviseSetAccessedBy = hip.chip.hipMemAdviseSetAccessedBy
    CU_MEM_ADVISE_UNSET_ACCESSED_BY = hip.chip.hipMemAdviseUnsetAccessedBy
    cudaMemAdviseUnsetAccessedBy = hip.chip.hipMemAdviseUnsetAccessedBy
    hipMemAdviseUnsetAccessedBy = hip.chip.hipMemAdviseUnsetAccessedBy
    hipMemAdviseSetCoarseGrain = hip.chip.hipMemAdviseSetCoarseGrain
    hipMemAdviseUnsetCoarseGrain = hip.chip.hipMemAdviseUnsetCoarseGrain
ctypedef CUmem_advise CUmem_advise_enum
ctypedef CUmem_advise cudaMemoryAdvise
cdef enum CUmem_range_attribute:
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = hip.chip.hipMemRangeAttributeReadMostly
    cudaMemRangeAttributeReadMostly = hip.chip.hipMemRangeAttributeReadMostly
    hipMemRangeAttributeReadMostly = hip.chip.hipMemRangeAttributeReadMostly
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = hip.chip.hipMemRangeAttributePreferredLocation
    cudaMemRangeAttributePreferredLocation = hip.chip.hipMemRangeAttributePreferredLocation
    hipMemRangeAttributePreferredLocation = hip.chip.hipMemRangeAttributePreferredLocation
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = hip.chip.hipMemRangeAttributeAccessedBy
    cudaMemRangeAttributeAccessedBy = hip.chip.hipMemRangeAttributeAccessedBy
    hipMemRangeAttributeAccessedBy = hip.chip.hipMemRangeAttributeAccessedBy
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = hip.chip.hipMemRangeAttributeLastPrefetchLocation
    cudaMemRangeAttributeLastPrefetchLocation = hip.chip.hipMemRangeAttributeLastPrefetchLocation
    hipMemRangeAttributeLastPrefetchLocation = hip.chip.hipMemRangeAttributeLastPrefetchLocation
    hipMemRangeAttributeCoherencyMode = hip.chip.hipMemRangeAttributeCoherencyMode
ctypedef CUmem_range_attribute CUmem_range_attribute_enum
ctypedef CUmem_range_attribute cudaMemRangeAttribute
cdef enum CUmemPool_attribute:
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = hip.chip.hipMemPoolReuseFollowEventDependencies
    cudaMemPoolReuseFollowEventDependencies = hip.chip.hipMemPoolReuseFollowEventDependencies
    hipMemPoolReuseFollowEventDependencies = hip.chip.hipMemPoolReuseFollowEventDependencies
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = hip.chip.hipMemPoolReuseAllowOpportunistic
    cudaMemPoolReuseAllowOpportunistic = hip.chip.hipMemPoolReuseAllowOpportunistic
    hipMemPoolReuseAllowOpportunistic = hip.chip.hipMemPoolReuseAllowOpportunistic
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = hip.chip.hipMemPoolReuseAllowInternalDependencies
    cudaMemPoolReuseAllowInternalDependencies = hip.chip.hipMemPoolReuseAllowInternalDependencies
    hipMemPoolReuseAllowInternalDependencies = hip.chip.hipMemPoolReuseAllowInternalDependencies
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = hip.chip.hipMemPoolAttrReleaseThreshold
    cudaMemPoolAttrReleaseThreshold = hip.chip.hipMemPoolAttrReleaseThreshold
    hipMemPoolAttrReleaseThreshold = hip.chip.hipMemPoolAttrReleaseThreshold
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = hip.chip.hipMemPoolAttrReservedMemCurrent
    cudaMemPoolAttrReservedMemCurrent = hip.chip.hipMemPoolAttrReservedMemCurrent
    hipMemPoolAttrReservedMemCurrent = hip.chip.hipMemPoolAttrReservedMemCurrent
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = hip.chip.hipMemPoolAttrReservedMemHigh
    cudaMemPoolAttrReservedMemHigh = hip.chip.hipMemPoolAttrReservedMemHigh
    hipMemPoolAttrReservedMemHigh = hip.chip.hipMemPoolAttrReservedMemHigh
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = hip.chip.hipMemPoolAttrUsedMemCurrent
    cudaMemPoolAttrUsedMemCurrent = hip.chip.hipMemPoolAttrUsedMemCurrent
    hipMemPoolAttrUsedMemCurrent = hip.chip.hipMemPoolAttrUsedMemCurrent
    CU_MEMPOOL_ATTR_USED_MEM_HIGH = hip.chip.hipMemPoolAttrUsedMemHigh
    cudaMemPoolAttrUsedMemHigh = hip.chip.hipMemPoolAttrUsedMemHigh
    hipMemPoolAttrUsedMemHigh = hip.chip.hipMemPoolAttrUsedMemHigh
ctypedef CUmemPool_attribute CUmemPool_attribute_enum
ctypedef CUmemPool_attribute cudaMemPoolAttr
cdef enum CUmemLocationType:
    CU_MEM_LOCATION_TYPE_INVALID = hip.chip.hipMemLocationTypeInvalid
    cudaMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    hipMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    CU_MEM_LOCATION_TYPE_DEVICE = hip.chip.hipMemLocationTypeDevice
    cudaMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice
    hipMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice
ctypedef CUmemLocationType CUmemLocationType_enum
ctypedef CUmemLocationType cudaMemLocationType
from hip.chip cimport hipMemLocation as CUmemLocation
from hip.chip cimport hipMemLocation as CUmemLocation_st
from hip.chip cimport hipMemLocation as CUmemLocation_v1
from hip.chip cimport hipMemLocation as cudaMemLocation
cdef enum CUmemAccess_flags:
    CU_MEM_ACCESS_FLAGS_PROT_NONE = hip.chip.hipMemAccessFlagsProtNone
    cudaMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    hipMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    CU_MEM_ACCESS_FLAGS_PROT_READ = hip.chip.hipMemAccessFlagsProtRead
    cudaMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    hipMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hip.chip.hipMemAccessFlagsProtReadWrite
    cudaMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite
    hipMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite
ctypedef CUmemAccess_flags CUmemAccess_flags_enum
ctypedef CUmemAccess_flags cudaMemAccessFlags
from hip.chip cimport hipMemAccessDesc as CUmemAccessDesc
from hip.chip cimport hipMemAccessDesc as CUmemAccessDesc_st
from hip.chip cimport hipMemAccessDesc as CUmemAccessDesc_v1
from hip.chip cimport hipMemAccessDesc as cudaMemAccessDesc
cdef enum CUmemAllocationType:
    CU_MEM_ALLOCATION_TYPE_INVALID = hip.chip.hipMemAllocationTypeInvalid
    cudaMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    hipMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    CU_MEM_ALLOCATION_TYPE_PINNED = hip.chip.hipMemAllocationTypePinned
    cudaMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    hipMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    CU_MEM_ALLOCATION_TYPE_MAX = hip.chip.hipMemAllocationTypeMax
    cudaMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax
    hipMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax
ctypedef CUmemAllocationType CUmemAllocationType_enum
ctypedef CUmemAllocationType cudaMemAllocationType
cdef enum CUmemAllocationHandleType:
    CU_MEM_HANDLE_TYPE_NONE = hip.chip.hipMemHandleTypeNone
    cudaMemHandleTypeNone = hip.chip.hipMemHandleTypeNone
    hipMemHandleTypeNone = hip.chip.hipMemHandleTypeNone
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = hip.chip.hipMemHandleTypePosixFileDescriptor
    cudaMemHandleTypePosixFileDescriptor = hip.chip.hipMemHandleTypePosixFileDescriptor
    hipMemHandleTypePosixFileDescriptor = hip.chip.hipMemHandleTypePosixFileDescriptor
    CU_MEM_HANDLE_TYPE_WIN32 = hip.chip.hipMemHandleTypeWin32
    cudaMemHandleTypeWin32 = hip.chip.hipMemHandleTypeWin32
    hipMemHandleTypeWin32 = hip.chip.hipMemHandleTypeWin32
    CU_MEM_HANDLE_TYPE_WIN32_KMT = hip.chip.hipMemHandleTypeWin32Kmt
    cudaMemHandleTypeWin32Kmt = hip.chip.hipMemHandleTypeWin32Kmt
    hipMemHandleTypeWin32Kmt = hip.chip.hipMemHandleTypeWin32Kmt
ctypedef CUmemAllocationHandleType CUmemAllocationHandleType_enum
ctypedef CUmemAllocationHandleType cudaMemAllocationHandleType
from hip.chip cimport hipMemPoolProps as CUmemPoolProps
from hip.chip cimport hipMemPoolProps as CUmemPoolProps_st
from hip.chip cimport hipMemPoolProps as CUmemPoolProps_v1
from hip.chip cimport hipMemPoolProps as cudaMemPoolProps
from hip.chip cimport hipMemPoolPtrExportData as CUmemPoolPtrExportData
from hip.chip cimport hipMemPoolPtrExportData as CUmemPoolPtrExportData_st
from hip.chip cimport hipMemPoolPtrExportData as CUmemPoolPtrExportData_v1
from hip.chip cimport hipMemPoolPtrExportData as cudaMemPoolPtrExportData
cdef enum CUjit_option:
    hipJitOptionMaxRegisters = hip.chip.hipJitOptionMaxRegisters
    hipJitOptionThreadsPerBlock = hip.chip.hipJitOptionThreadsPerBlock
    hipJitOptionWallTime = hip.chip.hipJitOptionWallTime
    hipJitOptionInfoLogBuffer = hip.chip.hipJitOptionInfoLogBuffer
    hipJitOptionInfoLogBufferSizeBytes = hip.chip.hipJitOptionInfoLogBufferSizeBytes
    hipJitOptionErrorLogBuffer = hip.chip.hipJitOptionErrorLogBuffer
    hipJitOptionErrorLogBufferSizeBytes = hip.chip.hipJitOptionErrorLogBufferSizeBytes
    hipJitOptionOptimizationLevel = hip.chip.hipJitOptionOptimizationLevel
    hipJitOptionTargetFromContext = hip.chip.hipJitOptionTargetFromContext
    hipJitOptionTarget = hip.chip.hipJitOptionTarget
    hipJitOptionFallbackStrategy = hip.chip.hipJitOptionFallbackStrategy
    hipJitOptionGenerateDebugInfo = hip.chip.hipJitOptionGenerateDebugInfo
    hipJitOptionLogVerbose = hip.chip.hipJitOptionLogVerbose
    hipJitOptionGenerateLineInfo = hip.chip.hipJitOptionGenerateLineInfo
    hipJitOptionCacheMode = hip.chip.hipJitOptionCacheMode
    hipJitOptionSm3xOpt = hip.chip.hipJitOptionSm3xOpt
    hipJitOptionFastCompile = hip.chip.hipJitOptionFastCompile
    hipJitOptionNumOptions = hip.chip.hipJitOptionNumOptions
ctypedef CUjit_option CUjit_option_enum
cdef enum cudaFuncAttribute:
    cudaFuncAttributeMaxDynamicSharedMemorySize = hip.chip.hipFuncAttributeMaxDynamicSharedMemorySize
    hipFuncAttributeMaxDynamicSharedMemorySize = hip.chip.hipFuncAttributeMaxDynamicSharedMemorySize
    cudaFuncAttributePreferredSharedMemoryCarveout = hip.chip.hipFuncAttributePreferredSharedMemoryCarveout
    hipFuncAttributePreferredSharedMemoryCarveout = hip.chip.hipFuncAttributePreferredSharedMemoryCarveout
    cudaFuncAttributeMax = hip.chip.hipFuncAttributeMax
    hipFuncAttributeMax = hip.chip.hipFuncAttributeMax
cdef enum CUfunc_cache:
    CU_FUNC_CACHE_PREFER_NONE = hip.chip.hipFuncCachePreferNone
    cudaFuncCachePreferNone = hip.chip.hipFuncCachePreferNone
    hipFuncCachePreferNone = hip.chip.hipFuncCachePreferNone
    CU_FUNC_CACHE_PREFER_SHARED = hip.chip.hipFuncCachePreferShared
    cudaFuncCachePreferShared = hip.chip.hipFuncCachePreferShared
    hipFuncCachePreferShared = hip.chip.hipFuncCachePreferShared
    CU_FUNC_CACHE_PREFER_L1 = hip.chip.hipFuncCachePreferL1
    cudaFuncCachePreferL1 = hip.chip.hipFuncCachePreferL1
    hipFuncCachePreferL1 = hip.chip.hipFuncCachePreferL1
    CU_FUNC_CACHE_PREFER_EQUAL = hip.chip.hipFuncCachePreferEqual
    cudaFuncCachePreferEqual = hip.chip.hipFuncCachePreferEqual
    hipFuncCachePreferEqual = hip.chip.hipFuncCachePreferEqual
ctypedef CUfunc_cache CUfunc_cache_enum
ctypedef CUfunc_cache cudaFuncCache
cdef enum CUsharedconfig:
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = hip.chip.hipSharedMemBankSizeDefault
    cudaSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    hipSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeFourByte
    cudaSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    hipSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeEightByte
    cudaSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
    hipSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
ctypedef CUsharedconfig CUsharedconfig_enum
ctypedef CUsharedconfig cudaSharedMemConfig
cdef enum CUexternalMemoryHandleType_enum:
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = hip.chip.hipExternalMemoryHandleTypeOpaqueFd
    cudaExternalMemoryHandleTypeOpaqueFd = hip.chip.hipExternalMemoryHandleTypeOpaqueFd
    hipExternalMemoryHandleTypeOpaqueFd = hip.chip.hipExternalMemoryHandleTypeOpaqueFd
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = hip.chip.hipExternalMemoryHandleTypeOpaqueWin32
    cudaExternalMemoryHandleTypeOpaqueWin32 = hip.chip.hipExternalMemoryHandleTypeOpaqueWin32
    hipExternalMemoryHandleTypeOpaqueWin32 = hip.chip.hipExternalMemoryHandleTypeOpaqueWin32
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = hip.chip.hipExternalMemoryHandleTypeOpaqueWin32Kmt
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt = hip.chip.hipExternalMemoryHandleTypeOpaqueWin32Kmt
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = hip.chip.hipExternalMemoryHandleTypeOpaqueWin32Kmt
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = hip.chip.hipExternalMemoryHandleTypeD3D12Heap
    cudaExternalMemoryHandleTypeD3D12Heap = hip.chip.hipExternalMemoryHandleTypeD3D12Heap
    hipExternalMemoryHandleTypeD3D12Heap = hip.chip.hipExternalMemoryHandleTypeD3D12Heap
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = hip.chip.hipExternalMemoryHandleTypeD3D12Resource
    cudaExternalMemoryHandleTypeD3D12Resource = hip.chip.hipExternalMemoryHandleTypeD3D12Resource
    hipExternalMemoryHandleTypeD3D12Resource = hip.chip.hipExternalMemoryHandleTypeD3D12Resource
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = hip.chip.hipExternalMemoryHandleTypeD3D11Resource
    cudaExternalMemoryHandleTypeD3D11Resource = hip.chip.hipExternalMemoryHandleTypeD3D11Resource
    hipExternalMemoryHandleTypeD3D11Resource = hip.chip.hipExternalMemoryHandleTypeD3D11Resource
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = hip.chip.hipExternalMemoryHandleTypeD3D11ResourceKmt
    cudaExternalMemoryHandleTypeD3D11ResourceKmt = hip.chip.hipExternalMemoryHandleTypeD3D11ResourceKmt
    hipExternalMemoryHandleTypeD3D11ResourceKmt = hip.chip.hipExternalMemoryHandleTypeD3D11ResourceKmt
ctypedef CUexternalMemoryHandleType_enum CUexternalMemoryHandleType
ctypedef CUexternalMemoryHandleType_enum cudaExternalMemoryHandleType
from hip.chip cimport hipExternalMemoryHandleDesc_st as CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
from hip.chip cimport hipExternalMemoryBufferDesc_st as CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
from hip.chip cimport hipExternalMemory_t as CUexternalMemory
from hip.chip cimport hipExternalMemory_t as cudaExternalMemory_t
cdef enum CUexternalSemaphoreHandleType_enum:
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = hip.chip.hipExternalSemaphoreHandleTypeOpaqueFd
    cudaExternalSemaphoreHandleTypeOpaqueFd = hip.chip.hipExternalSemaphoreHandleTypeOpaqueFd
    hipExternalSemaphoreHandleTypeOpaqueFd = hip.chip.hipExternalSemaphoreHandleTypeOpaqueFd
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = hip.chip.hipExternalSemaphoreHandleTypeOpaqueWin32
    cudaExternalSemaphoreHandleTypeOpaqueWin32 = hip.chip.hipExternalSemaphoreHandleTypeOpaqueWin32
    hipExternalSemaphoreHandleTypeOpaqueWin32 = hip.chip.hipExternalSemaphoreHandleTypeOpaqueWin32
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = hip.chip.hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = hip.chip.hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = hip.chip.hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = hip.chip.hipExternalSemaphoreHandleTypeD3D12Fence
    cudaExternalSemaphoreHandleTypeD3D12Fence = hip.chip.hipExternalSemaphoreHandleTypeD3D12Fence
    hipExternalSemaphoreHandleTypeD3D12Fence = hip.chip.hipExternalSemaphoreHandleTypeD3D12Fence
ctypedef CUexternalSemaphoreHandleType_enum CUexternalSemaphoreHandleType
ctypedef CUexternalSemaphoreHandleType_enum cudaExternalSemaphoreHandleType
from hip.chip cimport hipExternalSemaphoreHandleDesc_st as CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
from hip.chip cimport hipExternalSemaphore_t as CUexternalSemaphore
from hip.chip cimport hipExternalSemaphore_t as cudaExternalSemaphore_t
from hip.chip cimport hipExternalSemaphoreSignalParams_st as CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
from hip.chip cimport hipExternalSemaphoreWaitParams_st as CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
cdef enum CUGLDeviceList:
    CU_GL_DEVICE_LIST_ALL = hip.chip.hipGLDeviceListAll
    cudaGLDeviceListAll = hip.chip.hipGLDeviceListAll
    hipGLDeviceListAll = hip.chip.hipGLDeviceListAll
    CU_GL_DEVICE_LIST_CURRENT_FRAME = hip.chip.hipGLDeviceListCurrentFrame
    cudaGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    hipGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    CU_GL_DEVICE_LIST_NEXT_FRAME = hip.chip.hipGLDeviceListNextFrame
    cudaGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame
    hipGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame
ctypedef CUGLDeviceList CUGLDeviceList_enum
ctypedef CUGLDeviceList cudaGLDeviceList
cdef enum CUgraphicsRegisterFlags:
    CU_GRAPHICS_REGISTER_FLAGS_NONE = hip.chip.hipGraphicsRegisterFlagsNone
    cudaGraphicsRegisterFlagsNone = hip.chip.hipGraphicsRegisterFlagsNone
    hipGraphicsRegisterFlagsNone = hip.chip.hipGraphicsRegisterFlagsNone
    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = hip.chip.hipGraphicsRegisterFlagsReadOnly
    cudaGraphicsRegisterFlagsReadOnly = hip.chip.hipGraphicsRegisterFlagsReadOnly
    hipGraphicsRegisterFlagsReadOnly = hip.chip.hipGraphicsRegisterFlagsReadOnly
    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = hip.chip.hipGraphicsRegisterFlagsWriteDiscard
    cudaGraphicsRegisterFlagsWriteDiscard = hip.chip.hipGraphicsRegisterFlagsWriteDiscard
    hipGraphicsRegisterFlagsWriteDiscard = hip.chip.hipGraphicsRegisterFlagsWriteDiscard
    CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = hip.chip.hipGraphicsRegisterFlagsSurfaceLoadStore
    cudaGraphicsRegisterFlagsSurfaceLoadStore = hip.chip.hipGraphicsRegisterFlagsSurfaceLoadStore
    hipGraphicsRegisterFlagsSurfaceLoadStore = hip.chip.hipGraphicsRegisterFlagsSurfaceLoadStore
    CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = hip.chip.hipGraphicsRegisterFlagsTextureGather
    cudaGraphicsRegisterFlagsTextureGather = hip.chip.hipGraphicsRegisterFlagsTextureGather
    hipGraphicsRegisterFlagsTextureGather = hip.chip.hipGraphicsRegisterFlagsTextureGather
ctypedef CUgraphicsRegisterFlags CUgraphicsRegisterFlags_enum
ctypedef CUgraphicsRegisterFlags cudaGraphicsRegisterFlags
from hip.chip cimport ihipGraph as CUgraph_st
from hip.chip cimport hipGraphNode as CUgraphNode_st
from hip.chip cimport hipGraphExec as CUgraphExec_st
from hip.chip cimport hipUserObject as CUuserObject_st
cdef enum CUgraphNodeType:
    CU_GRAPH_NODE_TYPE_KERNEL = hip.chip.hipGraphNodeTypeKernel
    cudaGraphNodeTypeKernel = hip.chip.hipGraphNodeTypeKernel
    hipGraphNodeTypeKernel = hip.chip.hipGraphNodeTypeKernel
    CU_GRAPH_NODE_TYPE_MEMCPY = hip.chip.hipGraphNodeTypeMemcpy
    cudaGraphNodeTypeMemcpy = hip.chip.hipGraphNodeTypeMemcpy
    hipGraphNodeTypeMemcpy = hip.chip.hipGraphNodeTypeMemcpy
    CU_GRAPH_NODE_TYPE_MEMSET = hip.chip.hipGraphNodeTypeMemset
    cudaGraphNodeTypeMemset = hip.chip.hipGraphNodeTypeMemset
    hipGraphNodeTypeMemset = hip.chip.hipGraphNodeTypeMemset
    CU_GRAPH_NODE_TYPE_HOST = hip.chip.hipGraphNodeTypeHost
    cudaGraphNodeTypeHost = hip.chip.hipGraphNodeTypeHost
    hipGraphNodeTypeHost = hip.chip.hipGraphNodeTypeHost
    CU_GRAPH_NODE_TYPE_GRAPH = hip.chip.hipGraphNodeTypeGraph
    cudaGraphNodeTypeGraph = hip.chip.hipGraphNodeTypeGraph
    hipGraphNodeTypeGraph = hip.chip.hipGraphNodeTypeGraph
    CU_GRAPH_NODE_TYPE_EMPTY = hip.chip.hipGraphNodeTypeEmpty
    cudaGraphNodeTypeEmpty = hip.chip.hipGraphNodeTypeEmpty
    hipGraphNodeTypeEmpty = hip.chip.hipGraphNodeTypeEmpty
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = hip.chip.hipGraphNodeTypeWaitEvent
    cudaGraphNodeTypeWaitEvent = hip.chip.hipGraphNodeTypeWaitEvent
    hipGraphNodeTypeWaitEvent = hip.chip.hipGraphNodeTypeWaitEvent
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = hip.chip.hipGraphNodeTypeEventRecord
    cudaGraphNodeTypeEventRecord = hip.chip.hipGraphNodeTypeEventRecord
    hipGraphNodeTypeEventRecord = hip.chip.hipGraphNodeTypeEventRecord
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = hip.chip.hipGraphNodeTypeExtSemaphoreSignal
    cudaGraphNodeTypeExtSemaphoreSignal = hip.chip.hipGraphNodeTypeExtSemaphoreSignal
    hipGraphNodeTypeExtSemaphoreSignal = hip.chip.hipGraphNodeTypeExtSemaphoreSignal
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = hip.chip.hipGraphNodeTypeExtSemaphoreWait
    cudaGraphNodeTypeExtSemaphoreWait = hip.chip.hipGraphNodeTypeExtSemaphoreWait
    hipGraphNodeTypeExtSemaphoreWait = hip.chip.hipGraphNodeTypeExtSemaphoreWait
    hipGraphNodeTypeMemcpyFromSymbol = hip.chip.hipGraphNodeTypeMemcpyFromSymbol
    hipGraphNodeTypeMemcpyToSymbol = hip.chip.hipGraphNodeTypeMemcpyToSymbol
    CU_GRAPH_NODE_TYPE_COUNT = hip.chip.hipGraphNodeTypeCount
    cudaGraphNodeTypeCount = hip.chip.hipGraphNodeTypeCount
    hipGraphNodeTypeCount = hip.chip.hipGraphNodeTypeCount
ctypedef CUgraphNodeType CUgraphNodeType_enum
ctypedef CUgraphNodeType cudaGraphNodeType
from hip.chip cimport hipHostFn_t as CUhostFn
from hip.chip cimport hipHostFn_t as cudaHostFn_t
from hip.chip cimport hipHostNodeParams as CUDA_HOST_NODE_PARAMS
from hip.chip cimport hipHostNodeParams as CUDA_HOST_NODE_PARAMS_st
from hip.chip cimport hipHostNodeParams as CUDA_HOST_NODE_PARAMS_v1
from hip.chip cimport hipHostNodeParams as cudaHostNodeParams
from hip.chip cimport hipKernelNodeParams as CUDA_KERNEL_NODE_PARAMS
from hip.chip cimport hipKernelNodeParams as CUDA_KERNEL_NODE_PARAMS_st
from hip.chip cimport hipKernelNodeParams as CUDA_KERNEL_NODE_PARAMS_v1
from hip.chip cimport hipKernelNodeParams as cudaKernelNodeParams
from hip.chip cimport hipMemsetParams as CUDA_MEMSET_NODE_PARAMS
from hip.chip cimport hipMemsetParams as CUDA_MEMSET_NODE_PARAMS_st
from hip.chip cimport hipMemsetParams as CUDA_MEMSET_NODE_PARAMS_v1
from hip.chip cimport hipMemsetParams as cudaMemsetParams
cdef enum CUkernelNodeAttrID:
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    cudaKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    hipKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = hip.chip.hipKernelNodeAttributeCooperative
    cudaKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative
    hipKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative
ctypedef CUkernelNodeAttrID CUkernelNodeAttrID_enum
ctypedef CUkernelNodeAttrID cudaKernelNodeAttrID
cdef enum CUaccessProperty:
    CU_ACCESS_PROPERTY_NORMAL = hip.chip.hipAccessPropertyNormal
    cudaAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    hipAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    CU_ACCESS_PROPERTY_STREAMING = hip.chip.hipAccessPropertyStreaming
    cudaAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    hipAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    CU_ACCESS_PROPERTY_PERSISTING = hip.chip.hipAccessPropertyPersisting
    cudaAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting
    hipAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting
ctypedef CUaccessProperty CUaccessProperty_enum
ctypedef CUaccessProperty cudaAccessProperty
from hip.chip cimport hipAccessPolicyWindow as CUaccessPolicyWindow
from hip.chip cimport hipAccessPolicyWindow as CUaccessPolicyWindow_st
from hip.chip cimport hipAccessPolicyWindow as cudaAccessPolicyWindow
from hip.chip cimport hipKernelNodeAttrValue as CUkernelNodeAttrValue
from hip.chip cimport hipKernelNodeAttrValue as CUkernelNodeAttrValue_union
from hip.chip cimport hipKernelNodeAttrValue as CUkernelNodeAttrValue_v1
from hip.chip cimport hipKernelNodeAttrValue as cudaKernelNodeAttrValue
cdef enum CUgraphExecUpdateResult:
    CU_GRAPH_EXEC_UPDATE_SUCCESS = hip.chip.hipGraphExecUpdateSuccess
    cudaGraphExecUpdateSuccess = hip.chip.hipGraphExecUpdateSuccess
    hipGraphExecUpdateSuccess = hip.chip.hipGraphExecUpdateSuccess
    CU_GRAPH_EXEC_UPDATE_ERROR = hip.chip.hipGraphExecUpdateError
    cudaGraphExecUpdateError = hip.chip.hipGraphExecUpdateError
    hipGraphExecUpdateError = hip.chip.hipGraphExecUpdateError
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = hip.chip.hipGraphExecUpdateErrorTopologyChanged
    cudaGraphExecUpdateErrorTopologyChanged = hip.chip.hipGraphExecUpdateErrorTopologyChanged
    hipGraphExecUpdateErrorTopologyChanged = hip.chip.hipGraphExecUpdateErrorTopologyChanged
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = hip.chip.hipGraphExecUpdateErrorNodeTypeChanged
    cudaGraphExecUpdateErrorNodeTypeChanged = hip.chip.hipGraphExecUpdateErrorNodeTypeChanged
    hipGraphExecUpdateErrorNodeTypeChanged = hip.chip.hipGraphExecUpdateErrorNodeTypeChanged
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = hip.chip.hipGraphExecUpdateErrorFunctionChanged
    cudaGraphExecUpdateErrorFunctionChanged = hip.chip.hipGraphExecUpdateErrorFunctionChanged
    hipGraphExecUpdateErrorFunctionChanged = hip.chip.hipGraphExecUpdateErrorFunctionChanged
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = hip.chip.hipGraphExecUpdateErrorParametersChanged
    cudaGraphExecUpdateErrorParametersChanged = hip.chip.hipGraphExecUpdateErrorParametersChanged
    hipGraphExecUpdateErrorParametersChanged = hip.chip.hipGraphExecUpdateErrorParametersChanged
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = hip.chip.hipGraphExecUpdateErrorNotSupported
    cudaGraphExecUpdateErrorNotSupported = hip.chip.hipGraphExecUpdateErrorNotSupported
    hipGraphExecUpdateErrorNotSupported = hip.chip.hipGraphExecUpdateErrorNotSupported
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = hip.chip.hipGraphExecUpdateErrorUnsupportedFunctionChange
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = hip.chip.hipGraphExecUpdateErrorUnsupportedFunctionChange
    hipGraphExecUpdateErrorUnsupportedFunctionChange = hip.chip.hipGraphExecUpdateErrorUnsupportedFunctionChange
ctypedef CUgraphExecUpdateResult CUgraphExecUpdateResult_enum
ctypedef CUgraphExecUpdateResult cudaGraphExecUpdateResult
cdef enum CUstreamCaptureMode:
    CU_STREAM_CAPTURE_MODE_GLOBAL = hip.chip.hipStreamCaptureModeGlobal
    cudaStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    hipStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = hip.chip.hipStreamCaptureModeThreadLocal
    cudaStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    hipStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    CU_STREAM_CAPTURE_MODE_RELAXED = hip.chip.hipStreamCaptureModeRelaxed
    cudaStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed
    hipStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed
ctypedef CUstreamCaptureMode CUstreamCaptureMode_enum
ctypedef CUstreamCaptureMode cudaStreamCaptureMode
cdef enum CUstreamCaptureStatus:
    CU_STREAM_CAPTURE_STATUS_NONE = hip.chip.hipStreamCaptureStatusNone
    cudaStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    hipStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    CU_STREAM_CAPTURE_STATUS_ACTIVE = hip.chip.hipStreamCaptureStatusActive
    cudaStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    hipStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = hip.chip.hipStreamCaptureStatusInvalidated
    cudaStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated
    hipStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated
ctypedef CUstreamCaptureStatus CUstreamCaptureStatus_enum
ctypedef CUstreamCaptureStatus cudaStreamCaptureStatus
cdef enum CUstreamUpdateCaptureDependencies_flags:
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = hip.chip.hipStreamAddCaptureDependencies
    cudaStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    hipStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = hip.chip.hipStreamSetCaptureDependencies
    cudaStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies
    hipStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies
ctypedef CUstreamUpdateCaptureDependencies_flags CUstreamUpdateCaptureDependencies_flags_enum
ctypedef CUstreamUpdateCaptureDependencies_flags cudaStreamUpdateCaptureDependenciesFlags
cdef enum CUgraphMem_attribute:
    CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = hip.chip.hipGraphMemAttrUsedMemCurrent
    cudaGraphMemAttrUsedMemCurrent = hip.chip.hipGraphMemAttrUsedMemCurrent
    hipGraphMemAttrUsedMemCurrent = hip.chip.hipGraphMemAttrUsedMemCurrent
    CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = hip.chip.hipGraphMemAttrUsedMemHigh
    cudaGraphMemAttrUsedMemHigh = hip.chip.hipGraphMemAttrUsedMemHigh
    hipGraphMemAttrUsedMemHigh = hip.chip.hipGraphMemAttrUsedMemHigh
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = hip.chip.hipGraphMemAttrReservedMemCurrent
    cudaGraphMemAttrReservedMemCurrent = hip.chip.hipGraphMemAttrReservedMemCurrent
    hipGraphMemAttrReservedMemCurrent = hip.chip.hipGraphMemAttrReservedMemCurrent
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = hip.chip.hipGraphMemAttrReservedMemHigh
    cudaGraphMemAttrReservedMemHigh = hip.chip.hipGraphMemAttrReservedMemHigh
    hipGraphMemAttrReservedMemHigh = hip.chip.hipGraphMemAttrReservedMemHigh
ctypedef CUgraphMem_attribute CUgraphMem_attribute_enum
ctypedef CUgraphMem_attribute cudaGraphMemAttributeType
cdef enum CUuserObject_flags:
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = hip.chip.hipUserObjectNoDestructorSync
    cudaUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync
    hipUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync
ctypedef CUuserObject_flags CUuserObject_flags_enum
ctypedef CUuserObject_flags cudaUserObjectFlags
cdef enum CUuserObjectRetain_flags:
    CU_GRAPH_USER_OBJECT_MOVE = hip.chip.hipGraphUserObjectMove
    cudaGraphUserObjectMove = hip.chip.hipGraphUserObjectMove
    hipGraphUserObjectMove = hip.chip.hipGraphUserObjectMove
ctypedef CUuserObjectRetain_flags CUuserObjectRetain_flags_enum
ctypedef CUuserObjectRetain_flags cudaUserObjectRetainFlags
cdef enum CUgraphInstantiate_flags:
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    cudaGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    hipGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
ctypedef CUgraphInstantiate_flags CUgraphInstantiate_flags_enum
ctypedef CUgraphInstantiate_flags cudaGraphInstantiateFlags
from hip.chip cimport hipMemAllocationProp as CUmemAllocationProp
from hip.chip cimport hipMemAllocationProp as CUmemAllocationProp_st
from hip.chip cimport hipMemAllocationProp as CUmemAllocationProp_v1
cdef enum CUmemAllocationGranularity_flags:
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = hip.chip.hipMemAllocationGranularityMinimum
    hipMemAllocationGranularityMinimum = hip.chip.hipMemAllocationGranularityMinimum
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = hip.chip.hipMemAllocationGranularityRecommended
    hipMemAllocationGranularityRecommended = hip.chip.hipMemAllocationGranularityRecommended
ctypedef CUmemAllocationGranularity_flags CUmemAllocationGranularity_flags_enum
cdef enum CUmemHandleType:
    CU_MEM_HANDLE_TYPE_GENERIC = hip.chip.hipMemHandleTypeGeneric
    hipMemHandleTypeGeneric = hip.chip.hipMemHandleTypeGeneric
ctypedef CUmemHandleType CUmemHandleType_enum
cdef enum CUmemOperationType:
    CU_MEM_OPERATION_TYPE_MAP = hip.chip.hipMemOperationTypeMap
    hipMemOperationTypeMap = hip.chip.hipMemOperationTypeMap
    CU_MEM_OPERATION_TYPE_UNMAP = hip.chip.hipMemOperationTypeUnmap
    hipMemOperationTypeUnmap = hip.chip.hipMemOperationTypeUnmap
ctypedef CUmemOperationType CUmemOperationType_enum
cdef enum CUarraySparseSubresourceType:
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = hip.chip.hipArraySparseSubresourceTypeSparseLevel
    hipArraySparseSubresourceTypeSparseLevel = hip.chip.hipArraySparseSubresourceTypeSparseLevel
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = hip.chip.hipArraySparseSubresourceTypeMiptail
    hipArraySparseSubresourceTypeMiptail = hip.chip.hipArraySparseSubresourceTypeMiptail
ctypedef CUarraySparseSubresourceType CUarraySparseSubresourceType_enum
from hip.chip cimport hipArrayMapInfo as CUarrayMapInfo
from hip.chip cimport hipArrayMapInfo as CUarrayMapInfo_st
from hip.chip cimport hipArrayMapInfo as CUarrayMapInfo_v1
from hip.chip cimport hipInit as cuInit
from hip.chip cimport hipDriverGetVersion as cuDriverGetVersion
from hip.chip cimport hipDriverGetVersion as cudaDriverGetVersion
from hip.chip cimport hipRuntimeGetVersion as cudaRuntimeGetVersion
from hip.chip cimport hipDeviceGet as cuDeviceGet
from hip.chip cimport hipDeviceComputeCapability as cuDeviceComputeCapability
from hip.chip cimport hipDeviceGetName as cuDeviceGetName
from hip.chip cimport hipDeviceGetUuid as cuDeviceGetUuid
from hip.chip cimport hipDeviceGetUuid as cuDeviceGetUuid_v2
from hip.chip cimport hipDeviceGetP2PAttribute as cudaDeviceGetP2PAttribute
from hip.chip cimport hipDeviceGetP2PAttribute as cuDeviceGetP2PAttribute
from hip.chip cimport hipDeviceGetPCIBusId as cudaDeviceGetPCIBusId
from hip.chip cimport hipDeviceGetPCIBusId as cuDeviceGetPCIBusId
from hip.chip cimport hipDeviceGetByPCIBusId as cudaDeviceGetByPCIBusId
from hip.chip cimport hipDeviceGetByPCIBusId as cuDeviceGetByPCIBusId
from hip.chip cimport hipDeviceTotalMem as cuDeviceTotalMem
from hip.chip cimport hipDeviceTotalMem as cuDeviceTotalMem_v2
from hip.chip cimport hipDeviceSynchronize as cudaDeviceSynchronize
from hip.chip cimport hipDeviceSynchronize as cudaThreadSynchronize
from hip.chip cimport hipDeviceReset as cudaDeviceReset
from hip.chip cimport hipDeviceReset as cudaThreadExit
from hip.chip cimport hipSetDevice as cudaSetDevice
from hip.chip cimport hipGetDevice as cudaGetDevice
from hip.chip cimport hipGetDeviceCount as cuDeviceGetCount
from hip.chip cimport hipGetDeviceCount as cudaGetDeviceCount
from hip.chip cimport hipDeviceGetAttribute as cuDeviceGetAttribute
from hip.chip cimport hipDeviceGetAttribute as cudaDeviceGetAttribute
from hip.chip cimport hipDeviceGetDefaultMemPool as cuDeviceGetDefaultMemPool
from hip.chip cimport hipDeviceGetDefaultMemPool as cudaDeviceGetDefaultMemPool
from hip.chip cimport hipDeviceSetMemPool as cuDeviceSetMemPool
from hip.chip cimport hipDeviceSetMemPool as cudaDeviceSetMemPool
from hip.chip cimport hipDeviceGetMemPool as cuDeviceGetMemPool
from hip.chip cimport hipDeviceGetMemPool as cudaDeviceGetMemPool
from hip.chip cimport hipGetDeviceProperties as cudaGetDeviceProperties
from hip.chip cimport hipDeviceSetCacheConfig as cudaDeviceSetCacheConfig
from hip.chip cimport hipDeviceSetCacheConfig as cudaThreadSetCacheConfig
from hip.chip cimport hipDeviceGetCacheConfig as cudaDeviceGetCacheConfig
from hip.chip cimport hipDeviceGetCacheConfig as cudaThreadGetCacheConfig
from hip.chip cimport hipDeviceGetLimit as cudaDeviceGetLimit
from hip.chip cimport hipDeviceGetLimit as cuCtxGetLimit
from hip.chip cimport hipDeviceSetLimit as cudaDeviceSetLimit
from hip.chip cimport hipDeviceSetLimit as cuCtxSetLimit
from hip.chip cimport hipDeviceGetSharedMemConfig as cudaDeviceGetSharedMemConfig
from hip.chip cimport hipGetDeviceFlags as cudaGetDeviceFlags
from hip.chip cimport hipDeviceSetSharedMemConfig as cudaDeviceSetSharedMemConfig
from hip.chip cimport hipSetDeviceFlags as cudaSetDeviceFlags
from hip.chip cimport hipChooseDevice as cudaChooseDevice
from hip.chip cimport hipIpcGetMemHandle as cudaIpcGetMemHandle
from hip.chip cimport hipIpcGetMemHandle as cuIpcGetMemHandle
from hip.chip cimport hipIpcOpenMemHandle as cudaIpcOpenMemHandle
from hip.chip cimport hipIpcOpenMemHandle as cuIpcOpenMemHandle
from hip.chip cimport hipIpcCloseMemHandle as cudaIpcCloseMemHandle
from hip.chip cimport hipIpcCloseMemHandle as cuIpcCloseMemHandle
from hip.chip cimport hipIpcGetEventHandle as cudaIpcGetEventHandle
from hip.chip cimport hipIpcGetEventHandle as cuIpcGetEventHandle
from hip.chip cimport hipIpcOpenEventHandle as cudaIpcOpenEventHandle
from hip.chip cimport hipIpcOpenEventHandle as cuIpcOpenEventHandle
from hip.chip cimport hipFuncSetAttribute as cudaFuncSetAttribute
from hip.chip cimport hipFuncSetCacheConfig as cudaFuncSetCacheConfig
from hip.chip cimport hipFuncSetSharedMemConfig as cudaFuncSetSharedMemConfig
from hip.chip cimport hipGetLastError as cudaGetLastError
from hip.chip cimport hipPeekAtLastError as cudaPeekAtLastError
from hip.chip cimport hipGetErrorName as cudaGetErrorName
from hip.chip cimport hipGetErrorString as cudaGetErrorString
from hip.chip cimport hipDrvGetErrorName as cuGetErrorName
from hip.chip cimport hipDrvGetErrorString as cuGetErrorString
from hip.chip cimport hipStreamCreate as cudaStreamCreate
from hip.chip cimport hipStreamCreateWithFlags as cuStreamCreate
from hip.chip cimport hipStreamCreateWithFlags as cudaStreamCreateWithFlags
from hip.chip cimport hipStreamCreateWithPriority as cuStreamCreateWithPriority
from hip.chip cimport hipStreamCreateWithPriority as cudaStreamCreateWithPriority
from hip.chip cimport hipDeviceGetStreamPriorityRange as cudaDeviceGetStreamPriorityRange
from hip.chip cimport hipDeviceGetStreamPriorityRange as cuCtxGetStreamPriorityRange
from hip.chip cimport hipStreamDestroy as cuStreamDestroy
from hip.chip cimport hipStreamDestroy as cuStreamDestroy_v2
from hip.chip cimport hipStreamDestroy as cudaStreamDestroy
from hip.chip cimport hipStreamQuery as cuStreamQuery
from hip.chip cimport hipStreamQuery as cudaStreamQuery
from hip.chip cimport hipStreamSynchronize as cuStreamSynchronize
from hip.chip cimport hipStreamSynchronize as cudaStreamSynchronize
from hip.chip cimport hipStreamWaitEvent as cuStreamWaitEvent
from hip.chip cimport hipStreamWaitEvent as cudaStreamWaitEvent
from hip.chip cimport hipStreamGetFlags as cuStreamGetFlags
from hip.chip cimport hipStreamGetFlags as cudaStreamGetFlags
from hip.chip cimport hipStreamGetPriority as cuStreamGetPriority
from hip.chip cimport hipStreamGetPriority as cudaStreamGetPriority
from hip.chip cimport hipStreamCallback_t as CUstreamCallback
from hip.chip cimport hipStreamCallback_t as cudaStreamCallback_t
from hip.chip cimport hipStreamAddCallback as cuStreamAddCallback
from hip.chip cimport hipStreamAddCallback as cudaStreamAddCallback
from hip.chip cimport hipStreamWaitValue32 as cuStreamWaitValue32
from hip.chip cimport hipStreamWaitValue32 as cuStreamWaitValue32_v2
from hip.chip cimport hipStreamWaitValue64 as cuStreamWaitValue64
from hip.chip cimport hipStreamWaitValue64 as cuStreamWaitValue64_v2
from hip.chip cimport hipStreamWriteValue32 as cuStreamWriteValue32
from hip.chip cimport hipStreamWriteValue32 as cuStreamWriteValue32_v2
from hip.chip cimport hipStreamWriteValue64 as cuStreamWriteValue64
from hip.chip cimport hipStreamWriteValue64 as cuStreamWriteValue64_v2
from hip.chip cimport hipEventCreateWithFlags as cuEventCreate
from hip.chip cimport hipEventCreateWithFlags as cudaEventCreateWithFlags
from hip.chip cimport hipEventCreate as cudaEventCreate
from hip.chip cimport hipEventRecord as cuEventRecord
from hip.chip cimport hipEventRecord as cudaEventRecord
from hip.chip cimport hipEventDestroy as cuEventDestroy
from hip.chip cimport hipEventDestroy as cuEventDestroy_v2
from hip.chip cimport hipEventDestroy as cudaEventDestroy
from hip.chip cimport hipEventSynchronize as cuEventSynchronize
from hip.chip cimport hipEventSynchronize as cudaEventSynchronize
from hip.chip cimport hipEventElapsedTime as cuEventElapsedTime
from hip.chip cimport hipEventElapsedTime as cudaEventElapsedTime
from hip.chip cimport hipEventQuery as cuEventQuery
from hip.chip cimport hipEventQuery as cudaEventQuery
from hip.chip cimport hipPointerGetAttributes as cudaPointerGetAttributes
from hip.chip cimport hipPointerGetAttribute as cuPointerGetAttribute
from hip.chip cimport hipDrvPointerGetAttributes as cuPointerGetAttributes
from hip.chip cimport hipImportExternalSemaphore as cuImportExternalSemaphore
from hip.chip cimport hipImportExternalSemaphore as cudaImportExternalSemaphore
from hip.chip cimport hipSignalExternalSemaphoresAsync as cuSignalExternalSemaphoresAsync
from hip.chip cimport hipSignalExternalSemaphoresAsync as cudaSignalExternalSemaphoresAsync
from hip.chip cimport hipWaitExternalSemaphoresAsync as cuWaitExternalSemaphoresAsync
from hip.chip cimport hipWaitExternalSemaphoresAsync as cudaWaitExternalSemaphoresAsync
from hip.chip cimport hipDestroyExternalSemaphore as cuDestroyExternalSemaphore
from hip.chip cimport hipDestroyExternalSemaphore as cudaDestroyExternalSemaphore
from hip.chip cimport hipImportExternalMemory as cuImportExternalMemory
from hip.chip cimport hipImportExternalMemory as cudaImportExternalMemory
from hip.chip cimport hipExternalMemoryGetMappedBuffer as cuExternalMemoryGetMappedBuffer
from hip.chip cimport hipExternalMemoryGetMappedBuffer as cudaExternalMemoryGetMappedBuffer
from hip.chip cimport hipDestroyExternalMemory as cuDestroyExternalMemory
from hip.chip cimport hipDestroyExternalMemory as cudaDestroyExternalMemory
from hip.chip cimport hipMalloc as cuMemAlloc
from hip.chip cimport hipMalloc as cuMemAlloc_v2
from hip.chip cimport hipMalloc as cudaMalloc
from hip.chip cimport hipMemAllocHost as cuMemAllocHost
from hip.chip cimport hipMemAllocHost as cuMemAllocHost_v2
from hip.chip cimport hipHostMalloc as cudaMallocHost
from hip.chip cimport hipMallocManaged as cuMemAllocManaged
from hip.chip cimport hipMallocManaged as cudaMallocManaged
from hip.chip cimport hipMemPrefetchAsync as cudaMemPrefetchAsync
from hip.chip cimport hipMemPrefetchAsync as cuMemPrefetchAsync
from hip.chip cimport hipMemAdvise as cudaMemAdvise
from hip.chip cimport hipMemAdvise as cuMemAdvise
from hip.chip cimport hipMemRangeGetAttribute as cudaMemRangeGetAttribute
from hip.chip cimport hipMemRangeGetAttribute as cuMemRangeGetAttribute
from hip.chip cimport hipMemRangeGetAttributes as cudaMemRangeGetAttributes
from hip.chip cimport hipMemRangeGetAttributes as cuMemRangeGetAttributes
from hip.chip cimport hipStreamAttachMemAsync as cuStreamAttachMemAsync
from hip.chip cimport hipStreamAttachMemAsync as cudaStreamAttachMemAsync
from hip.chip cimport hipMallocAsync as cudaMallocAsync
from hip.chip cimport hipMallocAsync as cuMemAllocAsync
from hip.chip cimport hipFreeAsync as cudaFreeAsync
from hip.chip cimport hipFreeAsync as cuMemFreeAsync
from hip.chip cimport hipMemPoolTrimTo as cudaMemPoolTrimTo
from hip.chip cimport hipMemPoolTrimTo as cuMemPoolTrimTo
from hip.chip cimport hipMemPoolSetAttribute as cudaMemPoolSetAttribute
from hip.chip cimport hipMemPoolSetAttribute as cuMemPoolSetAttribute
from hip.chip cimport hipMemPoolGetAttribute as cudaMemPoolGetAttribute
from hip.chip cimport hipMemPoolGetAttribute as cuMemPoolGetAttribute
from hip.chip cimport hipMemPoolSetAccess as cudaMemPoolSetAccess
from hip.chip cimport hipMemPoolSetAccess as cuMemPoolSetAccess
from hip.chip cimport hipMemPoolGetAccess as cudaMemPoolGetAccess
from hip.chip cimport hipMemPoolGetAccess as cuMemPoolGetAccess
from hip.chip cimport hipMemPoolCreate as cudaMemPoolCreate
from hip.chip cimport hipMemPoolCreate as cuMemPoolCreate
from hip.chip cimport hipMemPoolDestroy as cudaMemPoolDestroy
from hip.chip cimport hipMemPoolDestroy as cuMemPoolDestroy
from hip.chip cimport hipMallocFromPoolAsync as cudaMallocFromPoolAsync
from hip.chip cimport hipMallocFromPoolAsync as cuMemAllocFromPoolAsync
from hip.chip cimport hipMemPoolExportToShareableHandle as cudaMemPoolExportToShareableHandle
from hip.chip cimport hipMemPoolExportToShareableHandle as cuMemPoolExportToShareableHandle
from hip.chip cimport hipMemPoolImportFromShareableHandle as cudaMemPoolImportFromShareableHandle
from hip.chip cimport hipMemPoolImportFromShareableHandle as cuMemPoolImportFromShareableHandle
from hip.chip cimport hipMemPoolExportPointer as cudaMemPoolExportPointer
from hip.chip cimport hipMemPoolExportPointer as cuMemPoolExportPointer
from hip.chip cimport hipMemPoolImportPointer as cudaMemPoolImportPointer
from hip.chip cimport hipMemPoolImportPointer as cuMemPoolImportPointer
from hip.chip cimport hipHostAlloc as cuMemHostAlloc
from hip.chip cimport hipHostAlloc as cudaHostAlloc
from hip.chip cimport hipHostGetDevicePointer as cuMemHostGetDevicePointer
from hip.chip cimport hipHostGetDevicePointer as cuMemHostGetDevicePointer_v2
from hip.chip cimport hipHostGetDevicePointer as cudaHostGetDevicePointer
from hip.chip cimport hipHostGetFlags as cuMemHostGetFlags
from hip.chip cimport hipHostGetFlags as cudaHostGetFlags
from hip.chip cimport hipHostRegister as cuMemHostRegister
from hip.chip cimport hipHostRegister as cuMemHostRegister_v2
from hip.chip cimport hipHostRegister as cudaHostRegister
from hip.chip cimport hipHostUnregister as cuMemHostUnregister
from hip.chip cimport hipHostUnregister as cudaHostUnregister
from hip.chip cimport hipMallocPitch as cudaMallocPitch
from hip.chip cimport hipMemAllocPitch as cuMemAllocPitch
from hip.chip cimport hipMemAllocPitch as cuMemAllocPitch_v2
from hip.chip cimport hipFree as cuMemFree
from hip.chip cimport hipFree as cuMemFree_v2
from hip.chip cimport hipFree as cudaFree
from hip.chip cimport hipHostFree as cuMemFreeHost
from hip.chip cimport hipHostFree as cudaFreeHost
from hip.chip cimport hipMemcpy as cudaMemcpy
from hip.chip cimport hipMemcpyHtoD as cuMemcpyHtoD
from hip.chip cimport hipMemcpyHtoD as cuMemcpyHtoD_v2
from hip.chip cimport hipMemcpyDtoH as cuMemcpyDtoH
from hip.chip cimport hipMemcpyDtoH as cuMemcpyDtoH_v2
from hip.chip cimport hipMemcpyDtoD as cuMemcpyDtoD
from hip.chip cimport hipMemcpyDtoD as cuMemcpyDtoD_v2
from hip.chip cimport hipMemcpyHtoDAsync as cuMemcpyHtoDAsync
from hip.chip cimport hipMemcpyHtoDAsync as cuMemcpyHtoDAsync_v2
from hip.chip cimport hipMemcpyDtoHAsync as cuMemcpyDtoHAsync
from hip.chip cimport hipMemcpyDtoHAsync as cuMemcpyDtoHAsync_v2
from hip.chip cimport hipMemcpyDtoDAsync as cuMemcpyDtoDAsync
from hip.chip cimport hipMemcpyDtoDAsync as cuMemcpyDtoDAsync_v2
from hip.chip cimport hipModuleGetGlobal as cuModuleGetGlobal
from hip.chip cimport hipModuleGetGlobal as cuModuleGetGlobal_v2
from hip.chip cimport hipGetSymbolAddress as cudaGetSymbolAddress
from hip.chip cimport hipGetSymbolSize as cudaGetSymbolSize
from hip.chip cimport hipMemcpyToSymbol as cudaMemcpyToSymbol
from hip.chip cimport hipMemcpyToSymbolAsync as cudaMemcpyToSymbolAsync
from hip.chip cimport hipMemcpyFromSymbol as cudaMemcpyFromSymbol
from hip.chip cimport hipMemcpyFromSymbolAsync as cudaMemcpyFromSymbolAsync
from hip.chip cimport hipMemcpyAsync as cudaMemcpyAsync
from hip.chip cimport hipMemset as cudaMemset
from hip.chip cimport hipMemsetD8 as cuMemsetD8
from hip.chip cimport hipMemsetD8 as cuMemsetD8_v2
from hip.chip cimport hipMemsetD8Async as cuMemsetD8Async
from hip.chip cimport hipMemsetD16 as cuMemsetD16
from hip.chip cimport hipMemsetD16 as cuMemsetD16_v2
from hip.chip cimport hipMemsetD16Async as cuMemsetD16Async
from hip.chip cimport hipMemsetD32 as cuMemsetD32
from hip.chip cimport hipMemsetD32 as cuMemsetD32_v2
from hip.chip cimport hipMemsetAsync as cudaMemsetAsync
from hip.chip cimport hipMemsetD32Async as cuMemsetD32Async
from hip.chip cimport hipMemset2D as cudaMemset2D
from hip.chip cimport hipMemset2DAsync as cudaMemset2DAsync
from hip.chip cimport hipMemset3D as cudaMemset3D
from hip.chip cimport hipMemset3DAsync as cudaMemset3DAsync
from hip.chip cimport hipMemGetInfo as cuMemGetInfo
from hip.chip cimport hipMemGetInfo as cuMemGetInfo_v2
from hip.chip cimport hipMemGetInfo as cudaMemGetInfo
from hip.chip cimport hipMallocArray as cudaMallocArray
from hip.chip cimport hipArrayCreate as cuArrayCreate
from hip.chip cimport hipArrayCreate as cuArrayCreate_v2
from hip.chip cimport hipArrayDestroy as cuArrayDestroy
from hip.chip cimport hipArray3DCreate as cuArray3DCreate
from hip.chip cimport hipArray3DCreate as cuArray3DCreate_v2
from hip.chip cimport hipMalloc3D as cudaMalloc3D
from hip.chip cimport hipFreeArray as cudaFreeArray
from hip.chip cimport hipFreeMipmappedArray as cudaFreeMipmappedArray
from hip.chip cimport hipMalloc3DArray as cudaMalloc3DArray
from hip.chip cimport hipMallocMipmappedArray as cudaMallocMipmappedArray
from hip.chip cimport hipGetMipmappedArrayLevel as cudaGetMipmappedArrayLevel
from hip.chip cimport hipMemcpy2D as cudaMemcpy2D
from hip.chip cimport hipMemcpyParam2D as cuMemcpy2D
from hip.chip cimport hipMemcpyParam2D as cuMemcpy2D_v2
from hip.chip cimport hipMemcpyParam2DAsync as cuMemcpy2DAsync
from hip.chip cimport hipMemcpyParam2DAsync as cuMemcpy2DAsync_v2
from hip.chip cimport hipMemcpy2DAsync as cudaMemcpy2DAsync
from hip.chip cimport hipMemcpy2DToArray as cudaMemcpy2DToArray
from hip.chip cimport hipMemcpy2DToArrayAsync as cudaMemcpy2DToArrayAsync
from hip.chip cimport hipMemcpyToArray as cudaMemcpyToArray
from hip.chip cimport hipMemcpyFromArray as cudaMemcpyFromArray
from hip.chip cimport hipMemcpy2DFromArray as cudaMemcpy2DFromArray
from hip.chip cimport hipMemcpy2DFromArrayAsync as cudaMemcpy2DFromArrayAsync
from hip.chip cimport hipMemcpyAtoH as cuMemcpyAtoH
from hip.chip cimport hipMemcpyAtoH as cuMemcpyAtoH_v2
from hip.chip cimport hipMemcpyHtoA as cuMemcpyHtoA
from hip.chip cimport hipMemcpyHtoA as cuMemcpyHtoA_v2
from hip.chip cimport hipMemcpy3D as cudaMemcpy3D
from hip.chip cimport hipMemcpy3DAsync as cudaMemcpy3DAsync
from hip.chip cimport hipDrvMemcpy3D as cuMemcpy3D
from hip.chip cimport hipDrvMemcpy3D as cuMemcpy3D_v2
from hip.chip cimport hipDrvMemcpy3DAsync as cuMemcpy3DAsync
from hip.chip cimport hipDrvMemcpy3DAsync as cuMemcpy3DAsync_v2
from hip.chip cimport hipDeviceCanAccessPeer as cuDeviceCanAccessPeer
from hip.chip cimport hipDeviceCanAccessPeer as cudaDeviceCanAccessPeer
from hip.chip cimport hipDeviceEnablePeerAccess as cudaDeviceEnablePeerAccess
from hip.chip cimport hipDeviceDisablePeerAccess as cudaDeviceDisablePeerAccess
from hip.chip cimport hipMemGetAddressRange as cuMemGetAddressRange
from hip.chip cimport hipMemGetAddressRange as cuMemGetAddressRange_v2
from hip.chip cimport hipMemcpyPeer as cudaMemcpyPeer
from hip.chip cimport hipMemcpyPeerAsync as cudaMemcpyPeerAsync
from hip.chip cimport hipCtxCreate as cuCtxCreate
from hip.chip cimport hipCtxCreate as cuCtxCreate_v2
from hip.chip cimport hipCtxDestroy as cuCtxDestroy
from hip.chip cimport hipCtxDestroy as cuCtxDestroy_v2
from hip.chip cimport hipCtxPopCurrent as cuCtxPopCurrent
from hip.chip cimport hipCtxPopCurrent as cuCtxPopCurrent_v2
from hip.chip cimport hipCtxPushCurrent as cuCtxPushCurrent
from hip.chip cimport hipCtxPushCurrent as cuCtxPushCurrent_v2
from hip.chip cimport hipCtxSetCurrent as cuCtxSetCurrent
from hip.chip cimport hipCtxGetCurrent as cuCtxGetCurrent
from hip.chip cimport hipCtxGetDevice as cuCtxGetDevice
from hip.chip cimport hipCtxGetApiVersion as cuCtxGetApiVersion
from hip.chip cimport hipCtxGetCacheConfig as cuCtxGetCacheConfig
from hip.chip cimport hipCtxSetCacheConfig as cuCtxSetCacheConfig
from hip.chip cimport hipCtxSetSharedMemConfig as cuCtxSetSharedMemConfig
from hip.chip cimport hipCtxGetSharedMemConfig as cuCtxGetSharedMemConfig
from hip.chip cimport hipCtxSynchronize as cuCtxSynchronize
from hip.chip cimport hipCtxGetFlags as cuCtxGetFlags
from hip.chip cimport hipCtxEnablePeerAccess as cuCtxEnablePeerAccess
from hip.chip cimport hipCtxDisablePeerAccess as cuCtxDisablePeerAccess
from hip.chip cimport hipDevicePrimaryCtxGetState as cuDevicePrimaryCtxGetState
from hip.chip cimport hipDevicePrimaryCtxRelease as cuDevicePrimaryCtxRelease
from hip.chip cimport hipDevicePrimaryCtxRelease as cuDevicePrimaryCtxRelease_v2
from hip.chip cimport hipDevicePrimaryCtxRetain as cuDevicePrimaryCtxRetain
from hip.chip cimport hipDevicePrimaryCtxReset as cuDevicePrimaryCtxReset
from hip.chip cimport hipDevicePrimaryCtxReset as cuDevicePrimaryCtxReset_v2
from hip.chip cimport hipDevicePrimaryCtxSetFlags as cuDevicePrimaryCtxSetFlags
from hip.chip cimport hipDevicePrimaryCtxSetFlags as cuDevicePrimaryCtxSetFlags_v2
from hip.chip cimport hipModuleLoad as cuModuleLoad
from hip.chip cimport hipModuleUnload as cuModuleUnload
from hip.chip cimport hipModuleGetFunction as cuModuleGetFunction
from hip.chip cimport hipFuncGetAttributes as cudaFuncGetAttributes
from hip.chip cimport hipFuncGetAttribute as cuFuncGetAttribute
from hip.chip cimport hipModuleGetTexRef as cuModuleGetTexRef
from hip.chip cimport hipModuleLoadData as cuModuleLoadData
from hip.chip cimport hipModuleLoadDataEx as cuModuleLoadDataEx
from hip.chip cimport hipModuleLaunchKernel as cuLaunchKernel
from hip.chip cimport hipLaunchCooperativeKernel as cudaLaunchCooperativeKernel
from hip.chip cimport hipLaunchCooperativeKernelMultiDevice as cudaLaunchCooperativeKernelMultiDevice
from hip.chip cimport hipModuleOccupancyMaxPotentialBlockSize as cuOccupancyMaxPotentialBlockSize
from hip.chip cimport hipModuleOccupancyMaxPotentialBlockSizeWithFlags as cuOccupancyMaxPotentialBlockSizeWithFlags
from hip.chip cimport hipModuleOccupancyMaxActiveBlocksPerMultiprocessor as cuOccupancyMaxActiveBlocksPerMultiprocessor
from hip.chip cimport hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags as cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
from hip.chip cimport hipOccupancyMaxActiveBlocksPerMultiprocessor as cudaOccupancyMaxActiveBlocksPerMultiprocessor
from hip.chip cimport hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags as cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
from hip.chip cimport hipOccupancyMaxPotentialBlockSize as cudaOccupancyMaxPotentialBlockSize
from hip.chip cimport hipProfilerStart as cuProfilerStart
from hip.chip cimport hipProfilerStart as cudaProfilerStart
from hip.chip cimport hipProfilerStop as cuProfilerStop
from hip.chip cimport hipProfilerStop as cudaProfilerStop
from hip.chip cimport hipConfigureCall as cudaConfigureCall
from hip.chip cimport hipSetupArgument as cudaSetupArgument
from hip.chip cimport hipLaunchByPtr as cudaLaunch
from hip.chip cimport hipLaunchKernel as cudaLaunchKernel
from hip.chip cimport hipLaunchHostFunc as cuLaunchHostFunc
from hip.chip cimport hipLaunchHostFunc as cudaLaunchHostFunc
from hip.chip cimport hipDrvMemcpy2DUnaligned as cuMemcpy2DUnaligned
from hip.chip cimport hipDrvMemcpy2DUnaligned as cuMemcpy2DUnaligned_v2
from hip.chip cimport hipBindTextureToMipmappedArray as cudaBindTextureToMipmappedArray
from hip.chip cimport hipCreateTextureObject as cudaCreateTextureObject
from hip.chip cimport hipDestroyTextureObject as cudaDestroyTextureObject
from hip.chip cimport hipGetChannelDesc as cudaGetChannelDesc
from hip.chip cimport hipGetTextureObjectResourceDesc as cudaGetTextureObjectResourceDesc
from hip.chip cimport hipGetTextureObjectResourceViewDesc as cudaGetTextureObjectResourceViewDesc
from hip.chip cimport hipGetTextureObjectTextureDesc as cudaGetTextureObjectTextureDesc
from hip.chip cimport hipTexObjectCreate as cuTexObjectCreate
from hip.chip cimport hipTexObjectDestroy as cuTexObjectDestroy
from hip.chip cimport hipTexObjectGetResourceDesc as cuTexObjectGetResourceDesc
from hip.chip cimport hipTexObjectGetResourceViewDesc as cuTexObjectGetResourceViewDesc
from hip.chip cimport hipTexObjectGetTextureDesc as cuTexObjectGetTextureDesc
from hip.chip cimport hipGetTextureReference as cudaGetTextureReference
from hip.chip cimport hipTexRefSetAddressMode as cuTexRefSetAddressMode
from hip.chip cimport hipTexRefSetArray as cuTexRefSetArray
from hip.chip cimport hipTexRefSetFilterMode as cuTexRefSetFilterMode
from hip.chip cimport hipTexRefSetFlags as cuTexRefSetFlags
from hip.chip cimport hipTexRefSetFormat as cuTexRefSetFormat
from hip.chip cimport hipBindTexture as cudaBindTexture
from hip.chip cimport hipBindTexture2D as cudaBindTexture2D
from hip.chip cimport hipBindTextureToArray as cudaBindTextureToArray
from hip.chip cimport hipGetTextureAlignmentOffset as cudaGetTextureAlignmentOffset
from hip.chip cimport hipUnbindTexture as cudaUnbindTexture
from hip.chip cimport hipTexRefGetAddress as cuTexRefGetAddress
from hip.chip cimport hipTexRefGetAddress as cuTexRefGetAddress_v2
from hip.chip cimport hipTexRefGetAddressMode as cuTexRefGetAddressMode
from hip.chip cimport hipTexRefGetFilterMode as cuTexRefGetFilterMode
from hip.chip cimport hipTexRefGetFlags as cuTexRefGetFlags
from hip.chip cimport hipTexRefGetFormat as cuTexRefGetFormat
from hip.chip cimport hipTexRefGetMaxAnisotropy as cuTexRefGetMaxAnisotropy
from hip.chip cimport hipTexRefGetMipmapFilterMode as cuTexRefGetMipmapFilterMode
from hip.chip cimport hipTexRefGetMipmapLevelBias as cuTexRefGetMipmapLevelBias
from hip.chip cimport hipTexRefGetMipmapLevelClamp as cuTexRefGetMipmapLevelClamp
from hip.chip cimport hipTexRefGetMipMappedArray as cuTexRefGetMipmappedArray
from hip.chip cimport hipTexRefSetAddress as cuTexRefSetAddress
from hip.chip cimport hipTexRefSetAddress as cuTexRefSetAddress_v2
from hip.chip cimport hipTexRefSetAddress2D as cuTexRefSetAddress2D
from hip.chip cimport hipTexRefSetAddress2D as cuTexRefSetAddress2D_v2
from hip.chip cimport hipTexRefSetAddress2D as cuTexRefSetAddress2D_v3
from hip.chip cimport hipTexRefSetMaxAnisotropy as cuTexRefSetMaxAnisotropy
from hip.chip cimport hipTexRefSetBorderColor as cuTexRefSetBorderColor
from hip.chip cimport hipTexRefSetMipmapFilterMode as cuTexRefSetMipmapFilterMode
from hip.chip cimport hipTexRefSetMipmapLevelBias as cuTexRefSetMipmapLevelBias
from hip.chip cimport hipTexRefSetMipmapLevelClamp as cuTexRefSetMipmapLevelClamp
from hip.chip cimport hipTexRefSetMipmappedArray as cuTexRefSetMipmappedArray
from hip.chip cimport hipMipmappedArrayCreate as cuMipmappedArrayCreate
from hip.chip cimport hipMipmappedArrayDestroy as cuMipmappedArrayDestroy
from hip.chip cimport hipMipmappedArrayGetLevel as cuMipmappedArrayGetLevel
from hip.chip cimport hipStreamBeginCapture as cuStreamBeginCapture
from hip.chip cimport hipStreamBeginCapture as cuStreamBeginCapture_v2
from hip.chip cimport hipStreamBeginCapture as cudaStreamBeginCapture
from hip.chip cimport hipStreamEndCapture as cuStreamEndCapture
from hip.chip cimport hipStreamEndCapture as cudaStreamEndCapture
from hip.chip cimport hipStreamGetCaptureInfo as cuStreamGetCaptureInfo
from hip.chip cimport hipStreamGetCaptureInfo as cudaStreamGetCaptureInfo
from hip.chip cimport hipStreamGetCaptureInfo_v2 as cuStreamGetCaptureInfo_v2
from hip.chip cimport hipStreamIsCapturing as cuStreamIsCapturing
from hip.chip cimport hipStreamIsCapturing as cudaStreamIsCapturing
from hip.chip cimport hipStreamUpdateCaptureDependencies as cuStreamUpdateCaptureDependencies
from hip.chip cimport hipThreadExchangeStreamCaptureMode as cuThreadExchangeStreamCaptureMode
from hip.chip cimport hipThreadExchangeStreamCaptureMode as cudaThreadExchangeStreamCaptureMode
from hip.chip cimport hipGraphCreate as cuGraphCreate
from hip.chip cimport hipGraphCreate as cudaGraphCreate
from hip.chip cimport hipGraphDestroy as cuGraphDestroy
from hip.chip cimport hipGraphDestroy as cudaGraphDestroy
from hip.chip cimport hipGraphAddDependencies as cuGraphAddDependencies
from hip.chip cimport hipGraphAddDependencies as cudaGraphAddDependencies
from hip.chip cimport hipGraphRemoveDependencies as cuGraphRemoveDependencies
from hip.chip cimport hipGraphRemoveDependencies as cudaGraphRemoveDependencies
from hip.chip cimport hipGraphGetEdges as cuGraphGetEdges
from hip.chip cimport hipGraphGetEdges as cudaGraphGetEdges
from hip.chip cimport hipGraphGetNodes as cuGraphGetNodes
from hip.chip cimport hipGraphGetNodes as cudaGraphGetNodes
from hip.chip cimport hipGraphGetRootNodes as cuGraphGetRootNodes
from hip.chip cimport hipGraphGetRootNodes as cudaGraphGetRootNodes
from hip.chip cimport hipGraphNodeGetDependencies as cuGraphNodeGetDependencies
from hip.chip cimport hipGraphNodeGetDependencies as cudaGraphNodeGetDependencies
from hip.chip cimport hipGraphNodeGetDependentNodes as cuGraphNodeGetDependentNodes
from hip.chip cimport hipGraphNodeGetDependentNodes as cudaGraphNodeGetDependentNodes
from hip.chip cimport hipGraphNodeGetType as cuGraphNodeGetType
from hip.chip cimport hipGraphNodeGetType as cudaGraphNodeGetType
from hip.chip cimport hipGraphDestroyNode as cuGraphDestroyNode
from hip.chip cimport hipGraphDestroyNode as cudaGraphDestroyNode
from hip.chip cimport hipGraphClone as cuGraphClone
from hip.chip cimport hipGraphClone as cudaGraphClone
from hip.chip cimport hipGraphNodeFindInClone as cuGraphNodeFindInClone
from hip.chip cimport hipGraphNodeFindInClone as cudaGraphNodeFindInClone
from hip.chip cimport hipGraphInstantiate as cuGraphInstantiate
from hip.chip cimport hipGraphInstantiate as cuGraphInstantiate_v2
from hip.chip cimport hipGraphInstantiate as cudaGraphInstantiate
from hip.chip cimport hipGraphInstantiateWithFlags as cuGraphInstantiateWithFlags
from hip.chip cimport hipGraphInstantiateWithFlags as cudaGraphInstantiateWithFlags
from hip.chip cimport hipGraphLaunch as cuGraphLaunch
from hip.chip cimport hipGraphLaunch as cudaGraphLaunch
from hip.chip cimport hipGraphUpload as cuGraphUpload
from hip.chip cimport hipGraphUpload as cudaGraphUpload
from hip.chip cimport hipGraphExecDestroy as cuGraphExecDestroy
from hip.chip cimport hipGraphExecDestroy as cudaGraphExecDestroy
from hip.chip cimport hipGraphExecUpdate as cuGraphExecUpdate
from hip.chip cimport hipGraphExecUpdate as cudaGraphExecUpdate
from hip.chip cimport hipGraphAddKernelNode as cuGraphAddKernelNode
from hip.chip cimport hipGraphAddKernelNode as cudaGraphAddKernelNode
from hip.chip cimport hipGraphKernelNodeGetParams as cuGraphKernelNodeGetParams
from hip.chip cimport hipGraphKernelNodeGetParams as cudaGraphKernelNodeGetParams
from hip.chip cimport hipGraphKernelNodeSetParams as cuGraphKernelNodeSetParams
from hip.chip cimport hipGraphKernelNodeSetParams as cudaGraphKernelNodeSetParams
from hip.chip cimport hipGraphExecKernelNodeSetParams as cuGraphExecKernelNodeSetParams
from hip.chip cimport hipGraphExecKernelNodeSetParams as cudaGraphExecKernelNodeSetParams
from hip.chip cimport hipGraphAddMemcpyNode as cudaGraphAddMemcpyNode
from hip.chip cimport hipGraphMemcpyNodeGetParams as cuGraphMemcpyNodeGetParams
from hip.chip cimport hipGraphMemcpyNodeGetParams as cudaGraphMemcpyNodeGetParams
from hip.chip cimport hipGraphMemcpyNodeSetParams as cuGraphMemcpyNodeSetParams
from hip.chip cimport hipGraphMemcpyNodeSetParams as cudaGraphMemcpyNodeSetParams
from hip.chip cimport hipGraphKernelNodeSetAttribute as cuGraphKernelNodeSetAttribute
from hip.chip cimport hipGraphKernelNodeSetAttribute as cudaGraphKernelNodeSetAttribute
from hip.chip cimport hipGraphKernelNodeGetAttribute as cuGraphKernelNodeGetAttribute
from hip.chip cimport hipGraphKernelNodeGetAttribute as cudaGraphKernelNodeGetAttribute
from hip.chip cimport hipGraphExecMemcpyNodeSetParams as cudaGraphExecMemcpyNodeSetParams
from hip.chip cimport hipGraphAddMemcpyNode1D as cudaGraphAddMemcpyNode1D
from hip.chip cimport hipGraphMemcpyNodeSetParams1D as cudaGraphMemcpyNodeSetParams1D
from hip.chip cimport hipGraphExecMemcpyNodeSetParams1D as cudaGraphExecMemcpyNodeSetParams1D
from hip.chip cimport hipGraphAddMemcpyNodeFromSymbol as cudaGraphAddMemcpyNodeFromSymbol
from hip.chip cimport hipGraphMemcpyNodeSetParamsFromSymbol as cudaGraphMemcpyNodeSetParamsFromSymbol
from hip.chip cimport hipGraphExecMemcpyNodeSetParamsFromSymbol as cudaGraphExecMemcpyNodeSetParamsFromSymbol
from hip.chip cimport hipGraphAddMemcpyNodeToSymbol as cudaGraphAddMemcpyNodeToSymbol
from hip.chip cimport hipGraphMemcpyNodeSetParamsToSymbol as cudaGraphMemcpyNodeSetParamsToSymbol
from hip.chip cimport hipGraphExecMemcpyNodeSetParamsToSymbol as cudaGraphExecMemcpyNodeSetParamsToSymbol
from hip.chip cimport hipGraphAddMemsetNode as cudaGraphAddMemsetNode
from hip.chip cimport hipGraphMemsetNodeGetParams as cuGraphMemsetNodeGetParams
from hip.chip cimport hipGraphMemsetNodeGetParams as cudaGraphMemsetNodeGetParams
from hip.chip cimport hipGraphMemsetNodeSetParams as cuGraphMemsetNodeSetParams
from hip.chip cimport hipGraphMemsetNodeSetParams as cudaGraphMemsetNodeSetParams
from hip.chip cimport hipGraphExecMemsetNodeSetParams as cudaGraphExecMemsetNodeSetParams
from hip.chip cimport hipGraphAddHostNode as cuGraphAddHostNode
from hip.chip cimport hipGraphAddHostNode as cudaGraphAddHostNode
from hip.chip cimport hipGraphHostNodeGetParams as cuGraphHostNodeGetParams
from hip.chip cimport hipGraphHostNodeGetParams as cudaGraphHostNodeGetParams
from hip.chip cimport hipGraphHostNodeSetParams as cuGraphHostNodeSetParams
from hip.chip cimport hipGraphHostNodeSetParams as cudaGraphHostNodeSetParams
from hip.chip cimport hipGraphExecHostNodeSetParams as cuGraphExecHostNodeSetParams
from hip.chip cimport hipGraphExecHostNodeSetParams as cudaGraphExecHostNodeSetParams
from hip.chip cimport hipGraphAddChildGraphNode as cuGraphAddChildGraphNode
from hip.chip cimport hipGraphAddChildGraphNode as cudaGraphAddChildGraphNode
from hip.chip cimport hipGraphChildGraphNodeGetGraph as cuGraphChildGraphNodeGetGraph
from hip.chip cimport hipGraphChildGraphNodeGetGraph as cudaGraphChildGraphNodeGetGraph
from hip.chip cimport hipGraphExecChildGraphNodeSetParams as cuGraphExecChildGraphNodeSetParams
from hip.chip cimport hipGraphExecChildGraphNodeSetParams as cudaGraphExecChildGraphNodeSetParams
from hip.chip cimport hipGraphAddEmptyNode as cuGraphAddEmptyNode
from hip.chip cimport hipGraphAddEmptyNode as cudaGraphAddEmptyNode
from hip.chip cimport hipGraphAddEventRecordNode as cuGraphAddEventRecordNode
from hip.chip cimport hipGraphAddEventRecordNode as cudaGraphAddEventRecordNode
from hip.chip cimport hipGraphEventRecordNodeGetEvent as cuGraphEventRecordNodeGetEvent
from hip.chip cimport hipGraphEventRecordNodeGetEvent as cudaGraphEventRecordNodeGetEvent
from hip.chip cimport hipGraphEventRecordNodeSetEvent as cuGraphEventRecordNodeSetEvent
from hip.chip cimport hipGraphEventRecordNodeSetEvent as cudaGraphEventRecordNodeSetEvent
from hip.chip cimport hipGraphExecEventRecordNodeSetEvent as cuGraphExecEventRecordNodeSetEvent
from hip.chip cimport hipGraphExecEventRecordNodeSetEvent as cudaGraphExecEventRecordNodeSetEvent
from hip.chip cimport hipGraphAddEventWaitNode as cuGraphAddEventWaitNode
from hip.chip cimport hipGraphAddEventWaitNode as cudaGraphAddEventWaitNode
from hip.chip cimport hipGraphEventWaitNodeGetEvent as cuGraphEventWaitNodeGetEvent
from hip.chip cimport hipGraphEventWaitNodeGetEvent as cudaGraphEventWaitNodeGetEvent
from hip.chip cimport hipGraphEventWaitNodeSetEvent as cuGraphEventWaitNodeSetEvent
from hip.chip cimport hipGraphEventWaitNodeSetEvent as cudaGraphEventWaitNodeSetEvent
from hip.chip cimport hipGraphExecEventWaitNodeSetEvent as cuGraphExecEventWaitNodeSetEvent
from hip.chip cimport hipGraphExecEventWaitNodeSetEvent as cudaGraphExecEventWaitNodeSetEvent
from hip.chip cimport hipDeviceGetGraphMemAttribute as cuDeviceGetGraphMemAttribute
from hip.chip cimport hipDeviceGetGraphMemAttribute as cudaDeviceGetGraphMemAttribute
from hip.chip cimport hipDeviceSetGraphMemAttribute as cuDeviceSetGraphMemAttribute
from hip.chip cimport hipDeviceSetGraphMemAttribute as cudaDeviceSetGraphMemAttribute
from hip.chip cimport hipDeviceGraphMemTrim as cuDeviceGraphMemTrim
from hip.chip cimport hipDeviceGraphMemTrim as cudaDeviceGraphMemTrim
from hip.chip cimport hipUserObjectCreate as cuUserObjectCreate
from hip.chip cimport hipUserObjectCreate as cudaUserObjectCreate
from hip.chip cimport hipUserObjectRelease as cuUserObjectRelease
from hip.chip cimport hipUserObjectRelease as cudaUserObjectRelease
from hip.chip cimport hipUserObjectRetain as cuUserObjectRetain
from hip.chip cimport hipUserObjectRetain as cudaUserObjectRetain
from hip.chip cimport hipGraphRetainUserObject as cuGraphRetainUserObject
from hip.chip cimport hipGraphRetainUserObject as cudaGraphRetainUserObject
from hip.chip cimport hipGraphReleaseUserObject as cuGraphReleaseUserObject
from hip.chip cimport hipGraphReleaseUserObject as cudaGraphReleaseUserObject
from hip.chip cimport hipMemAddressFree as cuMemAddressFree
from hip.chip cimport hipMemAddressReserve as cuMemAddressReserve
from hip.chip cimport hipMemCreate as cuMemCreate
from hip.chip cimport hipMemExportToShareableHandle as cuMemExportToShareableHandle
from hip.chip cimport hipMemGetAccess as cuMemGetAccess
from hip.chip cimport hipMemGetAllocationGranularity as cuMemGetAllocationGranularity
from hip.chip cimport hipMemGetAllocationPropertiesFromHandle as cuMemGetAllocationPropertiesFromHandle
from hip.chip cimport hipMemImportFromShareableHandle as cuMemImportFromShareableHandle
from hip.chip cimport hipMemMap as cuMemMap
from hip.chip cimport hipMemMapArrayAsync as cuMemMapArrayAsync
from hip.chip cimport hipMemRelease as cuMemRelease
from hip.chip cimport hipMemRetainAllocationHandle as cuMemRetainAllocationHandle
from hip.chip cimport hipMemSetAccess as cuMemSetAccess
from hip.chip cimport hipMemUnmap as cuMemUnmap
from hip.chip cimport GLuint as GLuint
from hip.chip cimport GLenum as GLenum
from hip.chip cimport hipGLGetDevices as cuGLGetDevices
from hip.chip cimport hipGLGetDevices as cudaGLGetDevices
from hip.chip cimport hipGraphicsGLRegisterBuffer as cuGraphicsGLRegisterBuffer
from hip.chip cimport hipGraphicsGLRegisterBuffer as cudaGraphicsGLRegisterBuffer
from hip.chip cimport hipGraphicsGLRegisterImage as cuGraphicsGLRegisterImage
from hip.chip cimport hipGraphicsGLRegisterImage as cudaGraphicsGLRegisterImage
from hip.chip cimport hipGraphicsMapResources as cuGraphicsMapResources
from hip.chip cimport hipGraphicsMapResources as cudaGraphicsMapResources
from hip.chip cimport hipGraphicsSubResourceGetMappedArray as cuGraphicsSubResourceGetMappedArray
from hip.chip cimport hipGraphicsSubResourceGetMappedArray as cudaGraphicsSubResourceGetMappedArray
from hip.chip cimport hipGraphicsResourceGetMappedPointer as cuGraphicsResourceGetMappedPointer
from hip.chip cimport hipGraphicsResourceGetMappedPointer as cuGraphicsResourceGetMappedPointer_v2
from hip.chip cimport hipGraphicsResourceGetMappedPointer as cudaGraphicsResourceGetMappedPointer
from hip.chip cimport hipGraphicsUnmapResources as cuGraphicsUnmapResources
from hip.chip cimport hipGraphicsUnmapResources as cudaGraphicsUnmapResources
from hip.chip cimport hipGraphicsUnregisterResource as cuGraphicsUnregisterResource
from hip.chip cimport hipGraphicsUnregisterResource as cudaGraphicsUnregisterResource