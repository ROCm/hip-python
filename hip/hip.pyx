# AMD_COPYRIGHT
from libc.stdint cimport *
import enum

from . cimport chip
HIP_VERSION_MAJOR = chip.HIP_VERSION_MAJOR

HIP_VERSION_MINOR = chip.HIP_VERSION_MINOR

HIP_VERSION_PATCH = chip.HIP_VERSION_PATCH

HIP_VERSION_BUILD_ID = chip.HIP_VERSION_BUILD_ID

HIP_VERSION = chip.HIP_VERSION

HIP_TRSA_OVERRIDE_FORMAT = chip.HIP_TRSA_OVERRIDE_FORMAT

HIP_TRSF_READ_AS_INTEGER = chip.HIP_TRSF_READ_AS_INTEGER

HIP_TRSF_NORMALIZED_COORDINATES = chip.HIP_TRSF_NORMALIZED_COORDINATES

HIP_TRSF_SRGB = chip.HIP_TRSF_SRGB

hipTextureType1D = chip.hipTextureType1D

hipTextureType2D = chip.hipTextureType2D

hipTextureType3D = chip.hipTextureType3D

hipTextureTypeCubemap = chip.hipTextureTypeCubemap

hipTextureType1DLayered = chip.hipTextureType1DLayered

hipTextureType2DLayered = chip.hipTextureType2DLayered

hipTextureTypeCubemapLayered = chip.hipTextureTypeCubemapLayered

HIP_IMAGE_OBJECT_SIZE_DWORD = chip.HIP_IMAGE_OBJECT_SIZE_DWORD

HIP_SAMPLER_OBJECT_SIZE_DWORD = chip.HIP_SAMPLER_OBJECT_SIZE_DWORD

HIP_SAMPLER_OBJECT_OFFSET_DWORD = chip.HIP_SAMPLER_OBJECT_OFFSET_DWORD

HIP_TEXTURE_OBJECT_SIZE_DWORD = chip.HIP_TEXTURE_OBJECT_SIZE_DWORD

hipIpcMemLazyEnablePeerAccess = chip.hipIpcMemLazyEnablePeerAccess

HIP_IPC_HANDLE_SIZE = chip.HIP_IPC_HANDLE_SIZE

hipStreamDefault = chip.hipStreamDefault

hipStreamNonBlocking = chip.hipStreamNonBlocking

hipEventDefault = chip.hipEventDefault

hipEventBlockingSync = chip.hipEventBlockingSync

hipEventDisableTiming = chip.hipEventDisableTiming

hipEventInterprocess = chip.hipEventInterprocess

hipEventReleaseToDevice = chip.hipEventReleaseToDevice

hipEventReleaseToSystem = chip.hipEventReleaseToSystem

hipHostMallocDefault = chip.hipHostMallocDefault

hipHostMallocPortable = chip.hipHostMallocPortable

hipHostMallocMapped = chip.hipHostMallocMapped

hipHostMallocWriteCombined = chip.hipHostMallocWriteCombined

hipHostMallocNumaUser = chip.hipHostMallocNumaUser

hipHostMallocCoherent = chip.hipHostMallocCoherent

hipHostMallocNonCoherent = chip.hipHostMallocNonCoherent

hipMemAttachGlobal = chip.hipMemAttachGlobal

hipMemAttachHost = chip.hipMemAttachHost

hipMemAttachSingle = chip.hipMemAttachSingle

hipDeviceMallocDefault = chip.hipDeviceMallocDefault

hipDeviceMallocFinegrained = chip.hipDeviceMallocFinegrained

hipMallocSignalMemory = chip.hipMallocSignalMemory

hipHostRegisterDefault = chip.hipHostRegisterDefault

hipHostRegisterPortable = chip.hipHostRegisterPortable

hipHostRegisterMapped = chip.hipHostRegisterMapped

hipHostRegisterIoMemory = chip.hipHostRegisterIoMemory

hipExtHostRegisterCoarseGrained = chip.hipExtHostRegisterCoarseGrained

hipDeviceScheduleAuto = chip.hipDeviceScheduleAuto

hipDeviceScheduleSpin = chip.hipDeviceScheduleSpin

hipDeviceScheduleYield = chip.hipDeviceScheduleYield

hipDeviceScheduleBlockingSync = chip.hipDeviceScheduleBlockingSync

hipDeviceScheduleMask = chip.hipDeviceScheduleMask

hipDeviceMapHost = chip.hipDeviceMapHost

hipDeviceLmemResizeToMax = chip.hipDeviceLmemResizeToMax

hipArrayDefault = chip.hipArrayDefault

hipArrayLayered = chip.hipArrayLayered

hipArraySurfaceLoadStore = chip.hipArraySurfaceLoadStore

hipArrayCubemap = chip.hipArrayCubemap

hipArrayTextureGather = chip.hipArrayTextureGather

hipOccupancyDefault = chip.hipOccupancyDefault

hipCooperativeLaunchMultiDeviceNoPreSync = chip.hipCooperativeLaunchMultiDeviceNoPreSync

hipCooperativeLaunchMultiDeviceNoPostSync = chip.hipCooperativeLaunchMultiDeviceNoPostSync

hipCpuDeviceId = chip.hipCpuDeviceId

hipInvalidDeviceId = chip.hipInvalidDeviceId

hipExtAnyOrderLaunch = chip.hipExtAnyOrderLaunch

hipStreamWaitValueGte = chip.hipStreamWaitValueGte

hipStreamWaitValueEq = chip.hipStreamWaitValueEq

hipStreamWaitValueAnd = chip.hipStreamWaitValueAnd

hipStreamWaitValueNor = chip.hipStreamWaitValueNor

HIP_SUCCESS = chip.HIP_SUCCESS
HIP_ERROR_INVALID_VALUE = chip.HIP_ERROR_INVALID_VALUE
HIP_ERROR_NOT_INITIALIZED = chip.HIP_ERROR_NOT_INITIALIZED
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = chip.HIP_ERROR_LAUNCH_OUT_OF_RESOURCES

class hipMemoryType(enum.IntEnum):
    hipMemoryTypeHost = chip.hipMemoryTypeHost
    hipMemoryTypeDevice = chip.hipMemoryTypeDevice
    hipMemoryTypeArray = chip.hipMemoryTypeArray
    hipMemoryTypeUnified = chip.hipMemoryTypeUnified
    hipMemoryTypeManaged = chip.hipMemoryTypeManaged

class hipError_t(enum.IntEnum):
    hipSuccess = chip.hipSuccess
    hipErrorInvalidValue = chip.hipErrorInvalidValue
    hipErrorOutOfMemory = chip.hipErrorOutOfMemory
    hipErrorMemoryAllocation = chip.hipErrorMemoryAllocation
    hipErrorNotInitialized = chip.hipErrorNotInitialized
    hipErrorInitializationError = chip.hipErrorInitializationError
    hipErrorDeinitialized = chip.hipErrorDeinitialized
    hipErrorProfilerDisabled = chip.hipErrorProfilerDisabled
    hipErrorProfilerNotInitialized = chip.hipErrorProfilerNotInitialized
    hipErrorProfilerAlreadyStarted = chip.hipErrorProfilerAlreadyStarted
    hipErrorProfilerAlreadyStopped = chip.hipErrorProfilerAlreadyStopped
    hipErrorInvalidConfiguration = chip.hipErrorInvalidConfiguration
    hipErrorInvalidPitchValue = chip.hipErrorInvalidPitchValue
    hipErrorInvalidSymbol = chip.hipErrorInvalidSymbol
    hipErrorInvalidDevicePointer = chip.hipErrorInvalidDevicePointer
    hipErrorInvalidMemcpyDirection = chip.hipErrorInvalidMemcpyDirection
    hipErrorInsufficientDriver = chip.hipErrorInsufficientDriver
    hipErrorMissingConfiguration = chip.hipErrorMissingConfiguration
    hipErrorPriorLaunchFailure = chip.hipErrorPriorLaunchFailure
    hipErrorInvalidDeviceFunction = chip.hipErrorInvalidDeviceFunction
    hipErrorNoDevice = chip.hipErrorNoDevice
    hipErrorInvalidDevice = chip.hipErrorInvalidDevice
    hipErrorInvalidImage = chip.hipErrorInvalidImage
    hipErrorInvalidContext = chip.hipErrorInvalidContext
    hipErrorContextAlreadyCurrent = chip.hipErrorContextAlreadyCurrent
    hipErrorMapFailed = chip.hipErrorMapFailed
    hipErrorMapBufferObjectFailed = chip.hipErrorMapBufferObjectFailed
    hipErrorUnmapFailed = chip.hipErrorUnmapFailed
    hipErrorArrayIsMapped = chip.hipErrorArrayIsMapped
    hipErrorAlreadyMapped = chip.hipErrorAlreadyMapped
    hipErrorNoBinaryForGpu = chip.hipErrorNoBinaryForGpu
    hipErrorAlreadyAcquired = chip.hipErrorAlreadyAcquired
    hipErrorNotMapped = chip.hipErrorNotMapped
    hipErrorNotMappedAsArray = chip.hipErrorNotMappedAsArray
    hipErrorNotMappedAsPointer = chip.hipErrorNotMappedAsPointer
    hipErrorECCNotCorrectable = chip.hipErrorECCNotCorrectable
    hipErrorUnsupportedLimit = chip.hipErrorUnsupportedLimit
    hipErrorContextAlreadyInUse = chip.hipErrorContextAlreadyInUse
    hipErrorPeerAccessUnsupported = chip.hipErrorPeerAccessUnsupported
    hipErrorInvalidKernelFile = chip.hipErrorInvalidKernelFile
    hipErrorInvalidGraphicsContext = chip.hipErrorInvalidGraphicsContext
    hipErrorInvalidSource = chip.hipErrorInvalidSource
    hipErrorFileNotFound = chip.hipErrorFileNotFound
    hipErrorSharedObjectSymbolNotFound = chip.hipErrorSharedObjectSymbolNotFound
    hipErrorSharedObjectInitFailed = chip.hipErrorSharedObjectInitFailed
    hipErrorOperatingSystem = chip.hipErrorOperatingSystem
    hipErrorInvalidHandle = chip.hipErrorInvalidHandle
    hipErrorInvalidResourceHandle = chip.hipErrorInvalidResourceHandle
    hipErrorIllegalState = chip.hipErrorIllegalState
    hipErrorNotFound = chip.hipErrorNotFound
    hipErrorNotReady = chip.hipErrorNotReady
    hipErrorIllegalAddress = chip.hipErrorIllegalAddress
    hipErrorLaunchOutOfResources = chip.hipErrorLaunchOutOfResources
    hipErrorLaunchTimeOut = chip.hipErrorLaunchTimeOut
    hipErrorPeerAccessAlreadyEnabled = chip.hipErrorPeerAccessAlreadyEnabled
    hipErrorPeerAccessNotEnabled = chip.hipErrorPeerAccessNotEnabled
    hipErrorSetOnActiveProcess = chip.hipErrorSetOnActiveProcess
    hipErrorContextIsDestroyed = chip.hipErrorContextIsDestroyed
    hipErrorAssert = chip.hipErrorAssert
    hipErrorHostMemoryAlreadyRegistered = chip.hipErrorHostMemoryAlreadyRegistered
    hipErrorHostMemoryNotRegistered = chip.hipErrorHostMemoryNotRegistered
    hipErrorLaunchFailure = chip.hipErrorLaunchFailure
    hipErrorCooperativeLaunchTooLarge = chip.hipErrorCooperativeLaunchTooLarge
    hipErrorNotSupported = chip.hipErrorNotSupported
    hipErrorStreamCaptureUnsupported = chip.hipErrorStreamCaptureUnsupported
    hipErrorStreamCaptureInvalidated = chip.hipErrorStreamCaptureInvalidated
    hipErrorStreamCaptureMerge = chip.hipErrorStreamCaptureMerge
    hipErrorStreamCaptureUnmatched = chip.hipErrorStreamCaptureUnmatched
    hipErrorStreamCaptureUnjoined = chip.hipErrorStreamCaptureUnjoined
    hipErrorStreamCaptureIsolation = chip.hipErrorStreamCaptureIsolation
    hipErrorStreamCaptureImplicit = chip.hipErrorStreamCaptureImplicit
    hipErrorCapturedEvent = chip.hipErrorCapturedEvent
    hipErrorStreamCaptureWrongThread = chip.hipErrorStreamCaptureWrongThread
    hipErrorGraphExecUpdateFailure = chip.hipErrorGraphExecUpdateFailure
    hipErrorUnknown = chip.hipErrorUnknown
    hipErrorRuntimeMemory = chip.hipErrorRuntimeMemory
    hipErrorRuntimeOther = chip.hipErrorRuntimeOther
    hipErrorTbd = chip.hipErrorTbd

class hipDeviceAttribute_t(enum.IntEnum):
    hipDeviceAttributeCudaCompatibleBegin = chip.hipDeviceAttributeCudaCompatibleBegin
    hipDeviceAttributeEccEnabled = chip.hipDeviceAttributeEccEnabled
    hipDeviceAttributeAccessPolicyMaxWindowSize = chip.hipDeviceAttributeAccessPolicyMaxWindowSize
    hipDeviceAttributeAsyncEngineCount = chip.hipDeviceAttributeAsyncEngineCount
    hipDeviceAttributeCanMapHostMemory = chip.hipDeviceAttributeCanMapHostMemory
    hipDeviceAttributeCanUseHostPointerForRegisteredMem = chip.hipDeviceAttributeCanUseHostPointerForRegisteredMem
    hipDeviceAttributeClockRate = chip.hipDeviceAttributeClockRate
    hipDeviceAttributeComputeMode = chip.hipDeviceAttributeComputeMode
    hipDeviceAttributeComputePreemptionSupported = chip.hipDeviceAttributeComputePreemptionSupported
    hipDeviceAttributeConcurrentKernels = chip.hipDeviceAttributeConcurrentKernels
    hipDeviceAttributeConcurrentManagedAccess = chip.hipDeviceAttributeConcurrentManagedAccess
    hipDeviceAttributeCooperativeLaunch = chip.hipDeviceAttributeCooperativeLaunch
    hipDeviceAttributeCooperativeMultiDeviceLaunch = chip.hipDeviceAttributeCooperativeMultiDeviceLaunch
    hipDeviceAttributeDeviceOverlap = chip.hipDeviceAttributeDeviceOverlap
    hipDeviceAttributeDirectManagedMemAccessFromHost = chip.hipDeviceAttributeDirectManagedMemAccessFromHost
    hipDeviceAttributeGlobalL1CacheSupported = chip.hipDeviceAttributeGlobalL1CacheSupported
    hipDeviceAttributeHostNativeAtomicSupported = chip.hipDeviceAttributeHostNativeAtomicSupported
    hipDeviceAttributeIntegrated = chip.hipDeviceAttributeIntegrated
    hipDeviceAttributeIsMultiGpuBoard = chip.hipDeviceAttributeIsMultiGpuBoard
    hipDeviceAttributeKernelExecTimeout = chip.hipDeviceAttributeKernelExecTimeout
    hipDeviceAttributeL2CacheSize = chip.hipDeviceAttributeL2CacheSize
    hipDeviceAttributeLocalL1CacheSupported = chip.hipDeviceAttributeLocalL1CacheSupported
    hipDeviceAttributeLuid = chip.hipDeviceAttributeLuid
    hipDeviceAttributeLuidDeviceNodeMask = chip.hipDeviceAttributeLuidDeviceNodeMask
    hipDeviceAttributeComputeCapabilityMajor = chip.hipDeviceAttributeComputeCapabilityMajor
    hipDeviceAttributeManagedMemory = chip.hipDeviceAttributeManagedMemory
    hipDeviceAttributeMaxBlocksPerMultiProcessor = chip.hipDeviceAttributeMaxBlocksPerMultiProcessor
    hipDeviceAttributeMaxBlockDimX = chip.hipDeviceAttributeMaxBlockDimX
    hipDeviceAttributeMaxBlockDimY = chip.hipDeviceAttributeMaxBlockDimY
    hipDeviceAttributeMaxBlockDimZ = chip.hipDeviceAttributeMaxBlockDimZ
    hipDeviceAttributeMaxGridDimX = chip.hipDeviceAttributeMaxGridDimX
    hipDeviceAttributeMaxGridDimY = chip.hipDeviceAttributeMaxGridDimY
    hipDeviceAttributeMaxGridDimZ = chip.hipDeviceAttributeMaxGridDimZ
    hipDeviceAttributeMaxSurface1D = chip.hipDeviceAttributeMaxSurface1D
    hipDeviceAttributeMaxSurface1DLayered = chip.hipDeviceAttributeMaxSurface1DLayered
    hipDeviceAttributeMaxSurface2D = chip.hipDeviceAttributeMaxSurface2D
    hipDeviceAttributeMaxSurface2DLayered = chip.hipDeviceAttributeMaxSurface2DLayered
    hipDeviceAttributeMaxSurface3D = chip.hipDeviceAttributeMaxSurface3D
    hipDeviceAttributeMaxSurfaceCubemap = chip.hipDeviceAttributeMaxSurfaceCubemap
    hipDeviceAttributeMaxSurfaceCubemapLayered = chip.hipDeviceAttributeMaxSurfaceCubemapLayered
    hipDeviceAttributeMaxTexture1DWidth = chip.hipDeviceAttributeMaxTexture1DWidth
    hipDeviceAttributeMaxTexture1DLayered = chip.hipDeviceAttributeMaxTexture1DLayered
    hipDeviceAttributeMaxTexture1DLinear = chip.hipDeviceAttributeMaxTexture1DLinear
    hipDeviceAttributeMaxTexture1DMipmap = chip.hipDeviceAttributeMaxTexture1DMipmap
    hipDeviceAttributeMaxTexture2DWidth = chip.hipDeviceAttributeMaxTexture2DWidth
    hipDeviceAttributeMaxTexture2DHeight = chip.hipDeviceAttributeMaxTexture2DHeight
    hipDeviceAttributeMaxTexture2DGather = chip.hipDeviceAttributeMaxTexture2DGather
    hipDeviceAttributeMaxTexture2DLayered = chip.hipDeviceAttributeMaxTexture2DLayered
    hipDeviceAttributeMaxTexture2DLinear = chip.hipDeviceAttributeMaxTexture2DLinear
    hipDeviceAttributeMaxTexture2DMipmap = chip.hipDeviceAttributeMaxTexture2DMipmap
    hipDeviceAttributeMaxTexture3DWidth = chip.hipDeviceAttributeMaxTexture3DWidth
    hipDeviceAttributeMaxTexture3DHeight = chip.hipDeviceAttributeMaxTexture3DHeight
    hipDeviceAttributeMaxTexture3DDepth = chip.hipDeviceAttributeMaxTexture3DDepth
    hipDeviceAttributeMaxTexture3DAlt = chip.hipDeviceAttributeMaxTexture3DAlt
    hipDeviceAttributeMaxTextureCubemap = chip.hipDeviceAttributeMaxTextureCubemap
    hipDeviceAttributeMaxTextureCubemapLayered = chip.hipDeviceAttributeMaxTextureCubemapLayered
    hipDeviceAttributeMaxThreadsDim = chip.hipDeviceAttributeMaxThreadsDim
    hipDeviceAttributeMaxThreadsPerBlock = chip.hipDeviceAttributeMaxThreadsPerBlock
    hipDeviceAttributeMaxThreadsPerMultiProcessor = chip.hipDeviceAttributeMaxThreadsPerMultiProcessor
    hipDeviceAttributeMaxPitch = chip.hipDeviceAttributeMaxPitch
    hipDeviceAttributeMemoryBusWidth = chip.hipDeviceAttributeMemoryBusWidth
    hipDeviceAttributeMemoryClockRate = chip.hipDeviceAttributeMemoryClockRate
    hipDeviceAttributeComputeCapabilityMinor = chip.hipDeviceAttributeComputeCapabilityMinor
    hipDeviceAttributeMultiGpuBoardGroupID = chip.hipDeviceAttributeMultiGpuBoardGroupID
    hipDeviceAttributeMultiprocessorCount = chip.hipDeviceAttributeMultiprocessorCount
    hipDeviceAttributeName = chip.hipDeviceAttributeName
    hipDeviceAttributePageableMemoryAccess = chip.hipDeviceAttributePageableMemoryAccess
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables = chip.hipDeviceAttributePageableMemoryAccessUsesHostPageTables
    hipDeviceAttributePciBusId = chip.hipDeviceAttributePciBusId
    hipDeviceAttributePciDeviceId = chip.hipDeviceAttributePciDeviceId
    hipDeviceAttributePciDomainID = chip.hipDeviceAttributePciDomainID
    hipDeviceAttributePersistingL2CacheMaxSize = chip.hipDeviceAttributePersistingL2CacheMaxSize
    hipDeviceAttributeMaxRegistersPerBlock = chip.hipDeviceAttributeMaxRegistersPerBlock
    hipDeviceAttributeMaxRegistersPerMultiprocessor = chip.hipDeviceAttributeMaxRegistersPerMultiprocessor
    hipDeviceAttributeReservedSharedMemPerBlock = chip.hipDeviceAttributeReservedSharedMemPerBlock
    hipDeviceAttributeMaxSharedMemoryPerBlock = chip.hipDeviceAttributeMaxSharedMemoryPerBlock
    hipDeviceAttributeSharedMemPerBlockOptin = chip.hipDeviceAttributeSharedMemPerBlockOptin
    hipDeviceAttributeSharedMemPerMultiprocessor = chip.hipDeviceAttributeSharedMemPerMultiprocessor
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio = chip.hipDeviceAttributeSingleToDoublePrecisionPerfRatio
    hipDeviceAttributeStreamPrioritiesSupported = chip.hipDeviceAttributeStreamPrioritiesSupported
    hipDeviceAttributeSurfaceAlignment = chip.hipDeviceAttributeSurfaceAlignment
    hipDeviceAttributeTccDriver = chip.hipDeviceAttributeTccDriver
    hipDeviceAttributeTextureAlignment = chip.hipDeviceAttributeTextureAlignment
    hipDeviceAttributeTexturePitchAlignment = chip.hipDeviceAttributeTexturePitchAlignment
    hipDeviceAttributeTotalConstantMemory = chip.hipDeviceAttributeTotalConstantMemory
    hipDeviceAttributeTotalGlobalMem = chip.hipDeviceAttributeTotalGlobalMem
    hipDeviceAttributeUnifiedAddressing = chip.hipDeviceAttributeUnifiedAddressing
    hipDeviceAttributeUuid = chip.hipDeviceAttributeUuid
    hipDeviceAttributeWarpSize = chip.hipDeviceAttributeWarpSize
    hipDeviceAttributeMemoryPoolsSupported = chip.hipDeviceAttributeMemoryPoolsSupported
    hipDeviceAttributeVirtualMemoryManagementSupported = chip.hipDeviceAttributeVirtualMemoryManagementSupported
    hipDeviceAttributeCudaCompatibleEnd = chip.hipDeviceAttributeCudaCompatibleEnd
    hipDeviceAttributeAmdSpecificBegin = chip.hipDeviceAttributeAmdSpecificBegin
    hipDeviceAttributeClockInstructionRate = chip.hipDeviceAttributeClockInstructionRate
    hipDeviceAttributeArch = chip.hipDeviceAttributeArch
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = chip.hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
    hipDeviceAttributeGcnArch = chip.hipDeviceAttributeGcnArch
    hipDeviceAttributeGcnArchName = chip.hipDeviceAttributeGcnArchName
    hipDeviceAttributeHdpMemFlushCntl = chip.hipDeviceAttributeHdpMemFlushCntl
    hipDeviceAttributeHdpRegFlushCntl = chip.hipDeviceAttributeHdpRegFlushCntl
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = chip.hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = chip.hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = chip.hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = chip.hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem
    hipDeviceAttributeIsLargeBar = chip.hipDeviceAttributeIsLargeBar
    hipDeviceAttributeAsicRevision = chip.hipDeviceAttributeAsicRevision
    hipDeviceAttributeCanUseStreamWaitValue = chip.hipDeviceAttributeCanUseStreamWaitValue
    hipDeviceAttributeImageSupport = chip.hipDeviceAttributeImageSupport
    hipDeviceAttributePhysicalMultiProcessorCount = chip.hipDeviceAttributePhysicalMultiProcessorCount
    hipDeviceAttributeFineGrainSupport = chip.hipDeviceAttributeFineGrainSupport
    hipDeviceAttributeWallClockRate = chip.hipDeviceAttributeWallClockRate
    hipDeviceAttributeAmdSpecificEnd = chip.hipDeviceAttributeAmdSpecificEnd
    hipDeviceAttributeVendorSpecificBegin = chip.hipDeviceAttributeVendorSpecificBegin

class hipComputeMode(enum.IntEnum):
    hipComputeModeDefault = chip.hipComputeModeDefault
    hipComputeModeExclusive = chip.hipComputeModeExclusive
    hipComputeModeProhibited = chip.hipComputeModeProhibited
    hipComputeModeExclusiveProcess = chip.hipComputeModeExclusiveProcess

class hipChannelFormatKind(enum.IntEnum):
    hipChannelFormatKindSigned = chip.hipChannelFormatKindSigned
    hipChannelFormatKindUnsigned = chip.hipChannelFormatKindUnsigned
    hipChannelFormatKindFloat = chip.hipChannelFormatKindFloat
    hipChannelFormatKindNone = chip.hipChannelFormatKindNone

class hipArray_Format(enum.IntEnum):
    HIP_AD_FORMAT_UNSIGNED_INT8 = chip.HIP_AD_FORMAT_UNSIGNED_INT8
    HIP_AD_FORMAT_UNSIGNED_INT16 = chip.HIP_AD_FORMAT_UNSIGNED_INT16
    HIP_AD_FORMAT_UNSIGNED_INT32 = chip.HIP_AD_FORMAT_UNSIGNED_INT32
    HIP_AD_FORMAT_SIGNED_INT8 = chip.HIP_AD_FORMAT_SIGNED_INT8
    HIP_AD_FORMAT_SIGNED_INT16 = chip.HIP_AD_FORMAT_SIGNED_INT16
    HIP_AD_FORMAT_SIGNED_INT32 = chip.HIP_AD_FORMAT_SIGNED_INT32
    HIP_AD_FORMAT_HALF = chip.HIP_AD_FORMAT_HALF
    HIP_AD_FORMAT_FLOAT = chip.HIP_AD_FORMAT_FLOAT

class hipResourceType(enum.IntEnum):
    hipResourceTypeArray = chip.hipResourceTypeArray
    hipResourceTypeMipmappedArray = chip.hipResourceTypeMipmappedArray
    hipResourceTypeLinear = chip.hipResourceTypeLinear
    hipResourceTypePitch2D = chip.hipResourceTypePitch2D

class HIPresourcetype_enum(enum.IntEnum):
    HIP_RESOURCE_TYPE_ARRAY = chip.HIP_RESOURCE_TYPE_ARRAY
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    HIP_RESOURCE_TYPE_LINEAR = chip.HIP_RESOURCE_TYPE_LINEAR
    HIP_RESOURCE_TYPE_PITCH2D = chip.HIP_RESOURCE_TYPE_PITCH2D

class HIPaddress_mode_enum(enum.IntEnum):
    HIP_TR_ADDRESS_MODE_WRAP = chip.HIP_TR_ADDRESS_MODE_WRAP
    HIP_TR_ADDRESS_MODE_CLAMP = chip.HIP_TR_ADDRESS_MODE_CLAMP
    HIP_TR_ADDRESS_MODE_MIRROR = chip.HIP_TR_ADDRESS_MODE_MIRROR
    HIP_TR_ADDRESS_MODE_BORDER = chip.HIP_TR_ADDRESS_MODE_BORDER

class HIPfilter_mode_enum(enum.IntEnum):
    HIP_TR_FILTER_MODE_POINT = chip.HIP_TR_FILTER_MODE_POINT
    HIP_TR_FILTER_MODE_LINEAR = chip.HIP_TR_FILTER_MODE_LINEAR

class hipResourceViewFormat(enum.IntEnum):
    hipResViewFormatNone = chip.hipResViewFormatNone
    hipResViewFormatUnsignedChar1 = chip.hipResViewFormatUnsignedChar1
    hipResViewFormatUnsignedChar2 = chip.hipResViewFormatUnsignedChar2
    hipResViewFormatUnsignedChar4 = chip.hipResViewFormatUnsignedChar4
    hipResViewFormatSignedChar1 = chip.hipResViewFormatSignedChar1
    hipResViewFormatSignedChar2 = chip.hipResViewFormatSignedChar2
    hipResViewFormatSignedChar4 = chip.hipResViewFormatSignedChar4
    hipResViewFormatUnsignedShort1 = chip.hipResViewFormatUnsignedShort1
    hipResViewFormatUnsignedShort2 = chip.hipResViewFormatUnsignedShort2
    hipResViewFormatUnsignedShort4 = chip.hipResViewFormatUnsignedShort4
    hipResViewFormatSignedShort1 = chip.hipResViewFormatSignedShort1
    hipResViewFormatSignedShort2 = chip.hipResViewFormatSignedShort2
    hipResViewFormatSignedShort4 = chip.hipResViewFormatSignedShort4
    hipResViewFormatUnsignedInt1 = chip.hipResViewFormatUnsignedInt1
    hipResViewFormatUnsignedInt2 = chip.hipResViewFormatUnsignedInt2
    hipResViewFormatUnsignedInt4 = chip.hipResViewFormatUnsignedInt4
    hipResViewFormatSignedInt1 = chip.hipResViewFormatSignedInt1
    hipResViewFormatSignedInt2 = chip.hipResViewFormatSignedInt2
    hipResViewFormatSignedInt4 = chip.hipResViewFormatSignedInt4
    hipResViewFormatHalf1 = chip.hipResViewFormatHalf1
    hipResViewFormatHalf2 = chip.hipResViewFormatHalf2
    hipResViewFormatHalf4 = chip.hipResViewFormatHalf4
    hipResViewFormatFloat1 = chip.hipResViewFormatFloat1
    hipResViewFormatFloat2 = chip.hipResViewFormatFloat2
    hipResViewFormatFloat4 = chip.hipResViewFormatFloat4
    hipResViewFormatUnsignedBlockCompressed1 = chip.hipResViewFormatUnsignedBlockCompressed1
    hipResViewFormatUnsignedBlockCompressed2 = chip.hipResViewFormatUnsignedBlockCompressed2
    hipResViewFormatUnsignedBlockCompressed3 = chip.hipResViewFormatUnsignedBlockCompressed3
    hipResViewFormatUnsignedBlockCompressed4 = chip.hipResViewFormatUnsignedBlockCompressed4
    hipResViewFormatSignedBlockCompressed4 = chip.hipResViewFormatSignedBlockCompressed4
    hipResViewFormatUnsignedBlockCompressed5 = chip.hipResViewFormatUnsignedBlockCompressed5
    hipResViewFormatSignedBlockCompressed5 = chip.hipResViewFormatSignedBlockCompressed5
    hipResViewFormatUnsignedBlockCompressed6H = chip.hipResViewFormatUnsignedBlockCompressed6H
    hipResViewFormatSignedBlockCompressed6H = chip.hipResViewFormatSignedBlockCompressed6H
    hipResViewFormatUnsignedBlockCompressed7 = chip.hipResViewFormatUnsignedBlockCompressed7

class HIPresourceViewFormat_enum(enum.IntEnum):
    HIP_RES_VIEW_FORMAT_NONE = chip.HIP_RES_VIEW_FORMAT_NONE
    HIP_RES_VIEW_FORMAT_UINT_1X8 = chip.HIP_RES_VIEW_FORMAT_UINT_1X8
    HIP_RES_VIEW_FORMAT_UINT_2X8 = chip.HIP_RES_VIEW_FORMAT_UINT_2X8
    HIP_RES_VIEW_FORMAT_UINT_4X8 = chip.HIP_RES_VIEW_FORMAT_UINT_4X8
    HIP_RES_VIEW_FORMAT_SINT_1X8 = chip.HIP_RES_VIEW_FORMAT_SINT_1X8
    HIP_RES_VIEW_FORMAT_SINT_2X8 = chip.HIP_RES_VIEW_FORMAT_SINT_2X8
    HIP_RES_VIEW_FORMAT_SINT_4X8 = chip.HIP_RES_VIEW_FORMAT_SINT_4X8
    HIP_RES_VIEW_FORMAT_UINT_1X16 = chip.HIP_RES_VIEW_FORMAT_UINT_1X16
    HIP_RES_VIEW_FORMAT_UINT_2X16 = chip.HIP_RES_VIEW_FORMAT_UINT_2X16
    HIP_RES_VIEW_FORMAT_UINT_4X16 = chip.HIP_RES_VIEW_FORMAT_UINT_4X16
    HIP_RES_VIEW_FORMAT_SINT_1X16 = chip.HIP_RES_VIEW_FORMAT_SINT_1X16
    HIP_RES_VIEW_FORMAT_SINT_2X16 = chip.HIP_RES_VIEW_FORMAT_SINT_2X16
    HIP_RES_VIEW_FORMAT_SINT_4X16 = chip.HIP_RES_VIEW_FORMAT_SINT_4X16
    HIP_RES_VIEW_FORMAT_UINT_1X32 = chip.HIP_RES_VIEW_FORMAT_UINT_1X32
    HIP_RES_VIEW_FORMAT_UINT_2X32 = chip.HIP_RES_VIEW_FORMAT_UINT_2X32
    HIP_RES_VIEW_FORMAT_UINT_4X32 = chip.HIP_RES_VIEW_FORMAT_UINT_4X32
    HIP_RES_VIEW_FORMAT_SINT_1X32 = chip.HIP_RES_VIEW_FORMAT_SINT_1X32
    HIP_RES_VIEW_FORMAT_SINT_2X32 = chip.HIP_RES_VIEW_FORMAT_SINT_2X32
    HIP_RES_VIEW_FORMAT_SINT_4X32 = chip.HIP_RES_VIEW_FORMAT_SINT_4X32
    HIP_RES_VIEW_FORMAT_FLOAT_1X16 = chip.HIP_RES_VIEW_FORMAT_FLOAT_1X16
    HIP_RES_VIEW_FORMAT_FLOAT_2X16 = chip.HIP_RES_VIEW_FORMAT_FLOAT_2X16
    HIP_RES_VIEW_FORMAT_FLOAT_4X16 = chip.HIP_RES_VIEW_FORMAT_FLOAT_4X16
    HIP_RES_VIEW_FORMAT_FLOAT_1X32 = chip.HIP_RES_VIEW_FORMAT_FLOAT_1X32
    HIP_RES_VIEW_FORMAT_FLOAT_2X32 = chip.HIP_RES_VIEW_FORMAT_FLOAT_2X32
    HIP_RES_VIEW_FORMAT_FLOAT_4X32 = chip.HIP_RES_VIEW_FORMAT_FLOAT_4X32
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
    HIP_RES_VIEW_FORMAT_SIGNED_BC4 = chip.HIP_RES_VIEW_FORMAT_SIGNED_BC4
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
    HIP_RES_VIEW_FORMAT_SIGNED_BC5 = chip.HIP_RES_VIEW_FORMAT_SIGNED_BC5
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H = chip.HIP_RES_VIEW_FORMAT_SIGNED_BC6H
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = chip.HIP_RES_VIEW_FORMAT_UNSIGNED_BC7

class hipMemcpyKind(enum.IntEnum):
    hipMemcpyHostToHost = chip.hipMemcpyHostToHost
    hipMemcpyHostToDevice = chip.hipMemcpyHostToDevice
    hipMemcpyDeviceToHost = chip.hipMemcpyDeviceToHost
    hipMemcpyDeviceToDevice = chip.hipMemcpyDeviceToDevice
    hipMemcpyDefault = chip.hipMemcpyDefault

class hipFunction_attribute(enum.IntEnum):
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = chip.HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = chip.HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = chip.HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = chip.HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
    HIP_FUNC_ATTRIBUTE_NUM_REGS = chip.HIP_FUNC_ATTRIBUTE_NUM_REGS
    HIP_FUNC_ATTRIBUTE_PTX_VERSION = chip.HIP_FUNC_ATTRIBUTE_PTX_VERSION
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION = chip.HIP_FUNC_ATTRIBUTE_BINARY_VERSION
    HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = chip.HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = chip.HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = chip.HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    HIP_FUNC_ATTRIBUTE_MAX = chip.HIP_FUNC_ATTRIBUTE_MAX

class hipPointer_attribute(enum.IntEnum):
    HIP_POINTER_ATTRIBUTE_CONTEXT = chip.HIP_POINTER_ATTRIBUTE_CONTEXT
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = chip.HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = chip.HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
    HIP_POINTER_ATTRIBUTE_HOST_POINTER = chip.HIP_POINTER_ATTRIBUTE_HOST_POINTER
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS = chip.HIP_POINTER_ATTRIBUTE_P2P_TOKENS
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = chip.HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
    HIP_POINTER_ATTRIBUTE_BUFFER_ID = chip.HIP_POINTER_ATTRIBUTE_BUFFER_ID
    HIP_POINTER_ATTRIBUTE_IS_MANAGED = chip.HIP_POINTER_ATTRIBUTE_IS_MANAGED
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = chip.HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = chip.HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = chip.HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE = chip.HIP_POINTER_ATTRIBUTE_RANGE_SIZE
    HIP_POINTER_ATTRIBUTE_MAPPED = chip.HIP_POINTER_ATTRIBUTE_MAPPED
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = chip.HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = chip.HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = chip.HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = chip.HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE

class hipTextureAddressMode(enum.IntEnum):
    hipAddressModeWrap = chip.hipAddressModeWrap
    hipAddressModeClamp = chip.hipAddressModeClamp
    hipAddressModeMirror = chip.hipAddressModeMirror
    hipAddressModeBorder = chip.hipAddressModeBorder

class hipTextureFilterMode(enum.IntEnum):
    hipFilterModePoint = chip.hipFilterModePoint
    hipFilterModeLinear = chip.hipFilterModeLinear

class hipTextureReadMode(enum.IntEnum):
    hipReadModeElementType = chip.hipReadModeElementType
    hipReadModeNormalizedFloat = chip.hipReadModeNormalizedFloat

class hipSurfaceBoundaryMode(enum.IntEnum):
    hipBoundaryModeZero = chip.hipBoundaryModeZero
    hipBoundaryModeTrap = chip.hipBoundaryModeTrap
    hipBoundaryModeClamp = chip.hipBoundaryModeClamp

class hipDeviceP2PAttr(enum.IntEnum):
    hipDevP2PAttrPerformanceRank = chip.hipDevP2PAttrPerformanceRank
    hipDevP2PAttrAccessSupported = chip.hipDevP2PAttrAccessSupported
    hipDevP2PAttrNativeAtomicSupported = chip.hipDevP2PAttrNativeAtomicSupported
    hipDevP2PAttrHipArrayAccessSupported = chip.hipDevP2PAttrHipArrayAccessSupported

class hipLimit_t(enum.IntEnum):
    hipLimitStackSize = chip.hipLimitStackSize
    hipLimitPrintfFifoSize = chip.hipLimitPrintfFifoSize
    hipLimitMallocHeapSize = chip.hipLimitMallocHeapSize
    hipLimitRange = chip.hipLimitRange

class hipMemoryAdvise(enum.IntEnum):
    hipMemAdviseSetReadMostly = chip.hipMemAdviseSetReadMostly
    hipMemAdviseUnsetReadMostly = chip.hipMemAdviseUnsetReadMostly
    hipMemAdviseSetPreferredLocation = chip.hipMemAdviseSetPreferredLocation
    hipMemAdviseUnsetPreferredLocation = chip.hipMemAdviseUnsetPreferredLocation
    hipMemAdviseSetAccessedBy = chip.hipMemAdviseSetAccessedBy
    hipMemAdviseUnsetAccessedBy = chip.hipMemAdviseUnsetAccessedBy
    hipMemAdviseSetCoarseGrain = chip.hipMemAdviseSetCoarseGrain
    hipMemAdviseUnsetCoarseGrain = chip.hipMemAdviseUnsetCoarseGrain

class hipMemRangeCoherencyMode(enum.IntEnum):
    hipMemRangeCoherencyModeFineGrain = chip.hipMemRangeCoherencyModeFineGrain
    hipMemRangeCoherencyModeCoarseGrain = chip.hipMemRangeCoherencyModeCoarseGrain
    hipMemRangeCoherencyModeIndeterminate = chip.hipMemRangeCoherencyModeIndeterminate

class hipMemRangeAttribute(enum.IntEnum):
    hipMemRangeAttributeReadMostly = chip.hipMemRangeAttributeReadMostly
    hipMemRangeAttributePreferredLocation = chip.hipMemRangeAttributePreferredLocation
    hipMemRangeAttributeAccessedBy = chip.hipMemRangeAttributeAccessedBy
    hipMemRangeAttributeLastPrefetchLocation = chip.hipMemRangeAttributeLastPrefetchLocation
    hipMemRangeAttributeCoherencyMode = chip.hipMemRangeAttributeCoherencyMode

class hipMemPoolAttr(enum.IntEnum):
    hipMemPoolReuseFollowEventDependencies = chip.hipMemPoolReuseFollowEventDependencies
    hipMemPoolReuseAllowOpportunistic = chip.hipMemPoolReuseAllowOpportunistic
    hipMemPoolReuseAllowInternalDependencies = chip.hipMemPoolReuseAllowInternalDependencies
    hipMemPoolAttrReleaseThreshold = chip.hipMemPoolAttrReleaseThreshold
    hipMemPoolAttrReservedMemCurrent = chip.hipMemPoolAttrReservedMemCurrent
    hipMemPoolAttrReservedMemHigh = chip.hipMemPoolAttrReservedMemHigh
    hipMemPoolAttrUsedMemCurrent = chip.hipMemPoolAttrUsedMemCurrent
    hipMemPoolAttrUsedMemHigh = chip.hipMemPoolAttrUsedMemHigh

class hipMemLocationType(enum.IntEnum):
    hipMemLocationTypeInvalid = chip.hipMemLocationTypeInvalid
    hipMemLocationTypeDevice = chip.hipMemLocationTypeDevice

class hipMemAccessFlags(enum.IntEnum):
    hipMemAccessFlagsProtNone = chip.hipMemAccessFlagsProtNone
    hipMemAccessFlagsProtRead = chip.hipMemAccessFlagsProtRead
    hipMemAccessFlagsProtReadWrite = chip.hipMemAccessFlagsProtReadWrite

class hipMemAllocationType(enum.IntEnum):
    hipMemAllocationTypeInvalid = chip.hipMemAllocationTypeInvalid
    hipMemAllocationTypePinned = chip.hipMemAllocationTypePinned
    hipMemAllocationTypeMax = chip.hipMemAllocationTypeMax

class hipMemAllocationHandleType(enum.IntEnum):
    hipMemHandleTypeNone = chip.hipMemHandleTypeNone
    hipMemHandleTypePosixFileDescriptor = chip.hipMemHandleTypePosixFileDescriptor
    hipMemHandleTypeWin32 = chip.hipMemHandleTypeWin32
    hipMemHandleTypeWin32Kmt = chip.hipMemHandleTypeWin32Kmt

class hipJitOption(enum.IntEnum):
    hipJitOptionMaxRegisters = chip.hipJitOptionMaxRegisters
    hipJitOptionThreadsPerBlock = chip.hipJitOptionThreadsPerBlock
    hipJitOptionWallTime = chip.hipJitOptionWallTime
    hipJitOptionInfoLogBuffer = chip.hipJitOptionInfoLogBuffer
    hipJitOptionInfoLogBufferSizeBytes = chip.hipJitOptionInfoLogBufferSizeBytes
    hipJitOptionErrorLogBuffer = chip.hipJitOptionErrorLogBuffer
    hipJitOptionErrorLogBufferSizeBytes = chip.hipJitOptionErrorLogBufferSizeBytes
    hipJitOptionOptimizationLevel = chip.hipJitOptionOptimizationLevel
    hipJitOptionTargetFromContext = chip.hipJitOptionTargetFromContext
    hipJitOptionTarget = chip.hipJitOptionTarget
    hipJitOptionFallbackStrategy = chip.hipJitOptionFallbackStrategy
    hipJitOptionGenerateDebugInfo = chip.hipJitOptionGenerateDebugInfo
    hipJitOptionLogVerbose = chip.hipJitOptionLogVerbose
    hipJitOptionGenerateLineInfo = chip.hipJitOptionGenerateLineInfo
    hipJitOptionCacheMode = chip.hipJitOptionCacheMode
    hipJitOptionSm3xOpt = chip.hipJitOptionSm3xOpt
    hipJitOptionFastCompile = chip.hipJitOptionFastCompile
    hipJitOptionNumOptions = chip.hipJitOptionNumOptions

class hipFuncAttribute(enum.IntEnum):
    hipFuncAttributeMaxDynamicSharedMemorySize = chip.hipFuncAttributeMaxDynamicSharedMemorySize
    hipFuncAttributePreferredSharedMemoryCarveout = chip.hipFuncAttributePreferredSharedMemoryCarveout
    hipFuncAttributeMax = chip.hipFuncAttributeMax

class hipFuncCache_t(enum.IntEnum):
    hipFuncCachePreferNone = chip.hipFuncCachePreferNone
    hipFuncCachePreferShared = chip.hipFuncCachePreferShared
    hipFuncCachePreferL1 = chip.hipFuncCachePreferL1
    hipFuncCachePreferEqual = chip.hipFuncCachePreferEqual

class hipSharedMemConfig(enum.IntEnum):
    hipSharedMemBankSizeDefault = chip.hipSharedMemBankSizeDefault
    hipSharedMemBankSizeFourByte = chip.hipSharedMemBankSizeFourByte
    hipSharedMemBankSizeEightByte = chip.hipSharedMemBankSizeEightByte

class hipExternalMemoryHandleType_enum(enum.IntEnum):
    hipExternalMemoryHandleTypeOpaqueFd = chip.hipExternalMemoryHandleTypeOpaqueFd
    hipExternalMemoryHandleTypeOpaqueWin32 = chip.hipExternalMemoryHandleTypeOpaqueWin32
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = chip.hipExternalMemoryHandleTypeOpaqueWin32Kmt
    hipExternalMemoryHandleTypeD3D12Heap = chip.hipExternalMemoryHandleTypeD3D12Heap
    hipExternalMemoryHandleTypeD3D12Resource = chip.hipExternalMemoryHandleTypeD3D12Resource
    hipExternalMemoryHandleTypeD3D11Resource = chip.hipExternalMemoryHandleTypeD3D11Resource
    hipExternalMemoryHandleTypeD3D11ResourceKmt = chip.hipExternalMemoryHandleTypeD3D11ResourceKmt

class hipExternalSemaphoreHandleType_enum(enum.IntEnum):
    hipExternalSemaphoreHandleTypeOpaqueFd = chip.hipExternalSemaphoreHandleTypeOpaqueFd
    hipExternalSemaphoreHandleTypeOpaqueWin32 = chip.hipExternalSemaphoreHandleTypeOpaqueWin32
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = chip.hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
    hipExternalSemaphoreHandleTypeD3D12Fence = chip.hipExternalSemaphoreHandleTypeD3D12Fence

class hipGLDeviceList(enum.IntEnum):
    hipGLDeviceListAll = chip.hipGLDeviceListAll
    hipGLDeviceListCurrentFrame = chip.hipGLDeviceListCurrentFrame
    hipGLDeviceListNextFrame = chip.hipGLDeviceListNextFrame

class hipGraphicsRegisterFlags(enum.IntEnum):
    hipGraphicsRegisterFlagsNone = chip.hipGraphicsRegisterFlagsNone
    hipGraphicsRegisterFlagsReadOnly = chip.hipGraphicsRegisterFlagsReadOnly
    hipGraphicsRegisterFlagsWriteDiscard = chip.hipGraphicsRegisterFlagsWriteDiscard
    hipGraphicsRegisterFlagsSurfaceLoadStore = chip.hipGraphicsRegisterFlagsSurfaceLoadStore
    hipGraphicsRegisterFlagsTextureGather = chip.hipGraphicsRegisterFlagsTextureGather

class hipGraphNodeType(enum.IntEnum):
    hipGraphNodeTypeKernel = chip.hipGraphNodeTypeKernel
    hipGraphNodeTypeMemcpy = chip.hipGraphNodeTypeMemcpy
    hipGraphNodeTypeMemset = chip.hipGraphNodeTypeMemset
    hipGraphNodeTypeHost = chip.hipGraphNodeTypeHost
    hipGraphNodeTypeGraph = chip.hipGraphNodeTypeGraph
    hipGraphNodeTypeEmpty = chip.hipGraphNodeTypeEmpty
    hipGraphNodeTypeWaitEvent = chip.hipGraphNodeTypeWaitEvent
    hipGraphNodeTypeEventRecord = chip.hipGraphNodeTypeEventRecord
    hipGraphNodeTypeExtSemaphoreSignal = chip.hipGraphNodeTypeExtSemaphoreSignal
    hipGraphNodeTypeExtSemaphoreWait = chip.hipGraphNodeTypeExtSemaphoreWait
    hipGraphNodeTypeMemcpyFromSymbol = chip.hipGraphNodeTypeMemcpyFromSymbol
    hipGraphNodeTypeMemcpyToSymbol = chip.hipGraphNodeTypeMemcpyToSymbol
    hipGraphNodeTypeCount = chip.hipGraphNodeTypeCount

class hipKernelNodeAttrID(enum.IntEnum):
    hipKernelNodeAttributeAccessPolicyWindow = chip.hipKernelNodeAttributeAccessPolicyWindow
    hipKernelNodeAttributeCooperative = chip.hipKernelNodeAttributeCooperative

class hipAccessProperty(enum.IntEnum):
    hipAccessPropertyNormal = chip.hipAccessPropertyNormal
    hipAccessPropertyStreaming = chip.hipAccessPropertyStreaming
    hipAccessPropertyPersisting = chip.hipAccessPropertyPersisting

class hipGraphExecUpdateResult(enum.IntEnum):
    hipGraphExecUpdateSuccess = chip.hipGraphExecUpdateSuccess
    hipGraphExecUpdateError = chip.hipGraphExecUpdateError
    hipGraphExecUpdateErrorTopologyChanged = chip.hipGraphExecUpdateErrorTopologyChanged
    hipGraphExecUpdateErrorNodeTypeChanged = chip.hipGraphExecUpdateErrorNodeTypeChanged
    hipGraphExecUpdateErrorFunctionChanged = chip.hipGraphExecUpdateErrorFunctionChanged
    hipGraphExecUpdateErrorParametersChanged = chip.hipGraphExecUpdateErrorParametersChanged
    hipGraphExecUpdateErrorNotSupported = chip.hipGraphExecUpdateErrorNotSupported
    hipGraphExecUpdateErrorUnsupportedFunctionChange = chip.hipGraphExecUpdateErrorUnsupportedFunctionChange

class hipStreamCaptureMode(enum.IntEnum):
    hipStreamCaptureModeGlobal = chip.hipStreamCaptureModeGlobal
    hipStreamCaptureModeThreadLocal = chip.hipStreamCaptureModeThreadLocal
    hipStreamCaptureModeRelaxed = chip.hipStreamCaptureModeRelaxed

class hipStreamCaptureStatus(enum.IntEnum):
    hipStreamCaptureStatusNone = chip.hipStreamCaptureStatusNone
    hipStreamCaptureStatusActive = chip.hipStreamCaptureStatusActive
    hipStreamCaptureStatusInvalidated = chip.hipStreamCaptureStatusInvalidated

class hipStreamUpdateCaptureDependenciesFlags(enum.IntEnum):
    hipStreamAddCaptureDependencies = chip.hipStreamAddCaptureDependencies
    hipStreamSetCaptureDependencies = chip.hipStreamSetCaptureDependencies

class hipGraphMemAttributeType(enum.IntEnum):
    hipGraphMemAttrUsedMemCurrent = chip.hipGraphMemAttrUsedMemCurrent
    hipGraphMemAttrUsedMemHigh = chip.hipGraphMemAttrUsedMemHigh
    hipGraphMemAttrReservedMemCurrent = chip.hipGraphMemAttrReservedMemCurrent
    hipGraphMemAttrReservedMemHigh = chip.hipGraphMemAttrReservedMemHigh

class hipUserObjectFlags(enum.IntEnum):
    hipUserObjectNoDestructorSync = chip.hipUserObjectNoDestructorSync

class hipUserObjectRetainFlags(enum.IntEnum):
    hipGraphUserObjectMove = chip.hipGraphUserObjectMove

class hipGraphInstantiateFlags(enum.IntEnum):
    hipGraphInstantiateFlagAutoFreeOnLaunch = chip.hipGraphInstantiateFlagAutoFreeOnLaunch

class hipMemAllocationGranularity_flags(enum.IntEnum):
    hipMemAllocationGranularityMinimum = chip.hipMemAllocationGranularityMinimum
    hipMemAllocationGranularityRecommended = chip.hipMemAllocationGranularityRecommended

class hipMemHandleType(enum.IntEnum):
    hipMemHandleTypeGeneric = chip.hipMemHandleTypeGeneric

class hipMemOperationType(enum.IntEnum):
    hipMemOperationTypeMap = chip.hipMemOperationTypeMap
    hipMemOperationTypeUnmap = chip.hipMemOperationTypeUnmap

class hipArraySparseSubresourceType(enum.IntEnum):
    hipArraySparseSubresourceTypeSparseLevel = chip.hipArraySparseSubresourceTypeSparseLevel
    hipArraySparseSubresourceTypeMiptail = chip.hipArraySparseSubresourceTypeMiptail