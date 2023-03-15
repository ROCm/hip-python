# AMD_COPYRIGHT
from . cimport chip

from chip cimport HIP_VERSION_MAJOR

from chip cimport HIP_VERSION_MINOR

from chip cimport HIP_VERSION_PATCH

from chip cimport HIP_VERSION_GITHASH

from chip cimport HIP_VERSION_BUILD_ID

from chip cimport HIP_VERSION_BUILD_NAME

from chip cimport HIP_VERSION

from chip cimport HIP_TRSA_OVERRIDE_FORMAT

from chip cimport HIP_TRSF_READ_AS_INTEGER

from chip cimport HIP_TRSF_NORMALIZED_COORDINATES

from chip cimport HIP_TRSF_SRGB

from chip cimport hipTextureType1D

from chip cimport hipTextureType2D

from chip cimport hipTextureType3D

from chip cimport hipTextureTypeCubemap

from chip cimport hipTextureType1DLayered

from chip cimport hipTextureType2DLayered

from chip cimport hipTextureTypeCubemapLayered

from chip cimport HIP_IMAGE_OBJECT_SIZE_DWORD

from chip cimport HIP_SAMPLER_OBJECT_SIZE_DWORD

from chip cimport HIP_SAMPLER_OBJECT_OFFSET_DWORD

from chip cimport HIP_TEXTURE_OBJECT_SIZE_DWORD

from chip cimport hipIpcMemLazyEnablePeerAccess

from chip cimport HIP_IPC_HANDLE_SIZE

from chip cimport hipStreamDefault

from chip cimport hipStreamNonBlocking

from chip cimport hipEventDefault

from chip cimport hipEventBlockingSync

from chip cimport hipEventDisableTiming

from chip cimport hipEventInterprocess

from chip cimport hipEventReleaseToDevice

from chip cimport hipEventReleaseToSystem

from chip cimport hipHostMallocDefault

from chip cimport hipHostMallocPortable

from chip cimport hipHostMallocMapped

from chip cimport hipHostMallocWriteCombined

from chip cimport hipHostMallocNumaUser

from chip cimport hipHostMallocCoherent

from chip cimport hipHostMallocNonCoherent

from chip cimport hipMemAttachGlobal

from chip cimport hipMemAttachHost

from chip cimport hipMemAttachSingle

from chip cimport hipDeviceMallocDefault

from chip cimport hipDeviceMallocFinegrained

from chip cimport hipMallocSignalMemory

from chip cimport hipHostRegisterDefault

from chip cimport hipHostRegisterPortable

from chip cimport hipHostRegisterMapped

from chip cimport hipHostRegisterIoMemory

from chip cimport hipExtHostRegisterCoarseGrained

from chip cimport hipDeviceScheduleAuto

from chip cimport hipDeviceScheduleSpin

from chip cimport hipDeviceScheduleYield

from chip cimport hipDeviceScheduleBlockingSync

from chip cimport hipDeviceScheduleMask

from chip cimport hipDeviceMapHost

from chip cimport hipDeviceLmemResizeToMax

from chip cimport hipArrayDefault

from chip cimport hipArrayLayered

from chip cimport hipArraySurfaceLoadStore

from chip cimport hipArrayCubemap

from chip cimport hipArrayTextureGather

from chip cimport hipOccupancyDefault

from chip cimport hipCooperativeLaunchMultiDeviceNoPreSync

from chip cimport hipCooperativeLaunchMultiDeviceNoPostSync

from chip cimport hipCpuDeviceId

from chip cimport hipInvalidDeviceId

from chip cimport hipExtAnyOrderLaunch

from chip cimport hipStreamWaitValueGte

from chip cimport hipStreamWaitValueEq

from chip cimport hipStreamWaitValueAnd

from chip cimport hipStreamWaitValueNor

from chip cimport hipStreamPerThread

class hipMemoryType(enum.IntEnum):
    hipMemoryTypeHost = 0
    hipMemoryTypeDevice = 1
    hipMemoryTypeArray = 2
    hipMemoryTypeUnified = 3
    hipMemoryTypeManaged = 4

class hipError_t(enum.IntEnum):
    hipSuccess = 0
    hipErrorInvalidValue = 1
    hipErrorOutOfMemory = 2
    hipErrorMemoryAllocation = 2
    hipErrorNotInitialized = 3
    hipErrorInitializationError = 3
    hipErrorDeinitialized = 4
    hipErrorProfilerDisabled = 5
    hipErrorProfilerNotInitialized = 6
    hipErrorProfilerAlreadyStarted = 7
    hipErrorProfilerAlreadyStopped = 8
    hipErrorInvalidConfiguration = 9
    hipErrorInvalidPitchValue = 12
    hipErrorInvalidSymbol = 13
    hipErrorInvalidDevicePointer = 17
    hipErrorInvalidMemcpyDirection = 21
    hipErrorInsufficientDriver = 35
    hipErrorMissingConfiguration = 52
    hipErrorPriorLaunchFailure = 53
    hipErrorInvalidDeviceFunction = 98
    hipErrorNoDevice = 100
    hipErrorInvalidDevice = 101
    hipErrorInvalidImage = 200
    hipErrorInvalidContext = 201
    hipErrorContextAlreadyCurrent = 202
    hipErrorMapFailed = 205
    hipErrorMapBufferObjectFailed = 205
    hipErrorUnmapFailed = 206
    hipErrorArrayIsMapped = 207
    hipErrorAlreadyMapped = 208
    hipErrorNoBinaryForGpu = 209
    hipErrorAlreadyAcquired = 210
    hipErrorNotMapped = 211
    hipErrorNotMappedAsArray = 212
    hipErrorNotMappedAsPointer = 213
    hipErrorECCNotCorrectable = 214
    hipErrorUnsupportedLimit = 215
    hipErrorContextAlreadyInUse = 216
    hipErrorPeerAccessUnsupported = 217
    hipErrorInvalidKernelFile = 218
    hipErrorInvalidGraphicsContext = 219
    hipErrorInvalidSource = 300
    hipErrorFileNotFound = 301
    hipErrorSharedObjectSymbolNotFound = 302
    hipErrorSharedObjectInitFailed = 303
    hipErrorOperatingSystem = 304
    hipErrorInvalidHandle = 400
    hipErrorInvalidResourceHandle = 400
    hipErrorIllegalState = 401
    hipErrorNotFound = 500
    hipErrorNotReady = 600
    hipErrorIllegalAddress = 700
    hipErrorLaunchOutOfResources = 701
    hipErrorLaunchTimeOut = 702
    hipErrorPeerAccessAlreadyEnabled = 704
    hipErrorPeerAccessNotEnabled = 705
    hipErrorSetOnActiveProcess = 708
    hipErrorContextIsDestroyed = 709
    hipErrorAssert = 710
    hipErrorHostMemoryAlreadyRegistered = 712
    hipErrorHostMemoryNotRegistered = 713
    hipErrorLaunchFailure = 719
    hipErrorCooperativeLaunchTooLarge = 720
    hipErrorNotSupported = 801
    hipErrorStreamCaptureUnsupported = 900
    hipErrorStreamCaptureInvalidated = 901
    hipErrorStreamCaptureMerge = 902
    hipErrorStreamCaptureUnmatched = 903
    hipErrorStreamCaptureUnjoined = 904
    hipErrorStreamCaptureIsolation = 905
    hipErrorStreamCaptureImplicit = 906
    hipErrorCapturedEvent = 907
    hipErrorStreamCaptureWrongThread = 908
    hipErrorGraphExecUpdateFailure = 910
    hipErrorUnknown = 999
    hipErrorRuntimeMemory = 1052
    hipErrorRuntimeOther = 1053
    hipErrorTbd = 1054

class hipDeviceAttribute_t(enum.IntEnum):
    hipDeviceAttributeCudaCompatibleBegin = 0
    hipDeviceAttributeEccEnabled = 0
    hipDeviceAttributeAccessPolicyMaxWindowSize = 1
    hipDeviceAttributeAsyncEngineCount = 2
    hipDeviceAttributeCanMapHostMemory = 3
    hipDeviceAttributeCanUseHostPointerForRegisteredMem = 4
    hipDeviceAttributeClockRate = 5
    hipDeviceAttributeComputeMode = 6
    hipDeviceAttributeComputePreemptionSupported = 7
    hipDeviceAttributeConcurrentKernels = 8
    hipDeviceAttributeConcurrentManagedAccess = 9
    hipDeviceAttributeCooperativeLaunch = 10
    hipDeviceAttributeCooperativeMultiDeviceLaunch = 11
    hipDeviceAttributeDeviceOverlap = 12
    hipDeviceAttributeDirectManagedMemAccessFromHost = 13
    hipDeviceAttributeGlobalL1CacheSupported = 14
    hipDeviceAttributeHostNativeAtomicSupported = 15
    hipDeviceAttributeIntegrated = 16
    hipDeviceAttributeIsMultiGpuBoard = 17
    hipDeviceAttributeKernelExecTimeout = 18
    hipDeviceAttributeL2CacheSize = 19
    hipDeviceAttributeLocalL1CacheSupported = 20
    hipDeviceAttributeLuid = 21
    hipDeviceAttributeLuidDeviceNodeMask = 22
    hipDeviceAttributeComputeCapabilityMajor = 23
    hipDeviceAttributeManagedMemory = 24
    hipDeviceAttributeMaxBlocksPerMultiProcessor = 25
    hipDeviceAttributeMaxBlockDimX = 26
    hipDeviceAttributeMaxBlockDimY = 27
    hipDeviceAttributeMaxBlockDimZ = 28
    hipDeviceAttributeMaxGridDimX = 29
    hipDeviceAttributeMaxGridDimY = 30
    hipDeviceAttributeMaxGridDimZ = 31
    hipDeviceAttributeMaxSurface1D = 32
    hipDeviceAttributeMaxSurface1DLayered = 33
    hipDeviceAttributeMaxSurface2D = 34
    hipDeviceAttributeMaxSurface2DLayered = 35
    hipDeviceAttributeMaxSurface3D = 36
    hipDeviceAttributeMaxSurfaceCubemap = 37
    hipDeviceAttributeMaxSurfaceCubemapLayered = 38
    hipDeviceAttributeMaxTexture1DWidth = 39
    hipDeviceAttributeMaxTexture1DLayered = 40
    hipDeviceAttributeMaxTexture1DLinear = 41
    hipDeviceAttributeMaxTexture1DMipmap = 42
    hipDeviceAttributeMaxTexture2DWidth = 43
    hipDeviceAttributeMaxTexture2DHeight = 44
    hipDeviceAttributeMaxTexture2DGather = 45
    hipDeviceAttributeMaxTexture2DLayered = 46
    hipDeviceAttributeMaxTexture2DLinear = 47
    hipDeviceAttributeMaxTexture2DMipmap = 48
    hipDeviceAttributeMaxTexture3DWidth = 49
    hipDeviceAttributeMaxTexture3DHeight = 50
    hipDeviceAttributeMaxTexture3DDepth = 51
    hipDeviceAttributeMaxTexture3DAlt = 52
    hipDeviceAttributeMaxTextureCubemap = 53
    hipDeviceAttributeMaxTextureCubemapLayered = 54
    hipDeviceAttributeMaxThreadsDim = 55
    hipDeviceAttributeMaxThreadsPerBlock = 56
    hipDeviceAttributeMaxThreadsPerMultiProcessor = 57
    hipDeviceAttributeMaxPitch = 58
    hipDeviceAttributeMemoryBusWidth = 59
    hipDeviceAttributeMemoryClockRate = 60
    hipDeviceAttributeComputeCapabilityMinor = 61
    hipDeviceAttributeMultiGpuBoardGroupID = 62
    hipDeviceAttributeMultiprocessorCount = 63
    hipDeviceAttributeName = 64
    hipDeviceAttributePageableMemoryAccess = 65
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 66
    hipDeviceAttributePciBusId = 67
    hipDeviceAttributePciDeviceId = 68
    hipDeviceAttributePciDomainID = 69
    hipDeviceAttributePersistingL2CacheMaxSize = 70
    hipDeviceAttributeMaxRegistersPerBlock = 71
    hipDeviceAttributeMaxRegistersPerMultiprocessor = 72
    hipDeviceAttributeReservedSharedMemPerBlock = 73
    hipDeviceAttributeMaxSharedMemoryPerBlock = 74
    hipDeviceAttributeSharedMemPerBlockOptin = 75
    hipDeviceAttributeSharedMemPerMultiprocessor = 76
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio = 77
    hipDeviceAttributeStreamPrioritiesSupported = 78
    hipDeviceAttributeSurfaceAlignment = 79
    hipDeviceAttributeTccDriver = 80
    hipDeviceAttributeTextureAlignment = 81
    hipDeviceAttributeTexturePitchAlignment = 82
    hipDeviceAttributeTotalConstantMemory = 83
    hipDeviceAttributeTotalGlobalMem = 84
    hipDeviceAttributeUnifiedAddressing = 85
    hipDeviceAttributeUuid = 86
    hipDeviceAttributeWarpSize = 87
    hipDeviceAttributeMemoryPoolsSupported = 88
    hipDeviceAttributeVirtualMemoryManagementSupported = 89
    hipDeviceAttributeCudaCompatibleEnd = 9999
    hipDeviceAttributeAmdSpecificBegin = 10000
    hipDeviceAttributeClockInstructionRate = 10000
    hipDeviceAttributeArch = 10001
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 10002
    hipDeviceAttributeGcnArch = 10003
    hipDeviceAttributeGcnArchName = 10004
    hipDeviceAttributeHdpMemFlushCntl = 10005
    hipDeviceAttributeHdpRegFlushCntl = 10006
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = 10007
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = 10008
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = 10009
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = 10010
    hipDeviceAttributeIsLargeBar = 10011
    hipDeviceAttributeAsicRevision = 10012
    hipDeviceAttributeCanUseStreamWaitValue = 10013
    hipDeviceAttributeImageSupport = 10014
    hipDeviceAttributePhysicalMultiProcessorCount = 10015
    hipDeviceAttributeFineGrainSupport = 10016
    hipDeviceAttributeWallClockRate = 10017
    hipDeviceAttributeAmdSpecificEnd = 19999
    hipDeviceAttributeVendorSpecificBegin = 20000

class hipComputeMode(enum.IntEnum):
    hipComputeModeDefault = 0
    hipComputeModeExclusive = 1
    hipComputeModeProhibited = 2
    hipComputeModeExclusiveProcess = 3

class hipChannelFormatKind(enum.IntEnum):
    hipChannelFormatKindSigned = 0
    hipChannelFormatKindUnsigned = 1
    hipChannelFormatKindFloat = 2
    hipChannelFormatKindNone = 3

class hipArray_Format(enum.IntEnum):
    HIP_AD_FORMAT_UNSIGNED_INT8 = 1
    HIP_AD_FORMAT_UNSIGNED_INT16 = 2
    HIP_AD_FORMAT_UNSIGNED_INT32 = 3
    HIP_AD_FORMAT_SIGNED_INT8 = 8
    HIP_AD_FORMAT_SIGNED_INT16 = 9
    HIP_AD_FORMAT_SIGNED_INT32 = 10
    HIP_AD_FORMAT_HALF = 16
    HIP_AD_FORMAT_FLOAT = 32

class hipResourceType(enum.IntEnum):
    hipResourceTypeArray = 0
    hipResourceTypeMipmappedArray = 1
    hipResourceTypeLinear = 2
    hipResourceTypePitch2D = 3

class HIPresourcetype_enum(enum.IntEnum):
    HIP_RESOURCE_TYPE_ARRAY = 0
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    HIP_RESOURCE_TYPE_LINEAR = 2
    HIP_RESOURCE_TYPE_PITCH2D = 3

class HIPresourcetype(enum.IntEnum):
    HIP_RESOURCE_TYPE_ARRAY = 0
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    HIP_RESOURCE_TYPE_LINEAR = 2
    HIP_RESOURCE_TYPE_PITCH2D = 3

class hipResourcetype(enum.IntEnum):
    HIP_RESOURCE_TYPE_ARRAY = 0
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    HIP_RESOURCE_TYPE_LINEAR = 2
    HIP_RESOURCE_TYPE_PITCH2D = 3

class HIPaddress_mode_enum(enum.IntEnum):
    HIP_TR_ADDRESS_MODE_WRAP = 0
    HIP_TR_ADDRESS_MODE_CLAMP = 1
    HIP_TR_ADDRESS_MODE_MIRROR = 2
    HIP_TR_ADDRESS_MODE_BORDER = 3

class HIPaddress_mode(enum.IntEnum):
    HIP_TR_ADDRESS_MODE_WRAP = 0
    HIP_TR_ADDRESS_MODE_CLAMP = 1
    HIP_TR_ADDRESS_MODE_MIRROR = 2
    HIP_TR_ADDRESS_MODE_BORDER = 3

class HIPfilter_mode_enum(enum.IntEnum):
    HIP_TR_FILTER_MODE_POINT = 0
    HIP_TR_FILTER_MODE_LINEAR = 1

class HIPfilter_mode(enum.IntEnum):
    HIP_TR_FILTER_MODE_POINT = 0
    HIP_TR_FILTER_MODE_LINEAR = 1

class hipResourceViewFormat(enum.IntEnum):
    hipResViewFormatNone = 0
    hipResViewFormatUnsignedChar1 = 1
    hipResViewFormatUnsignedChar2 = 2
    hipResViewFormatUnsignedChar4 = 3
    hipResViewFormatSignedChar1 = 4
    hipResViewFormatSignedChar2 = 5
    hipResViewFormatSignedChar4 = 6
    hipResViewFormatUnsignedShort1 = 7
    hipResViewFormatUnsignedShort2 = 8
    hipResViewFormatUnsignedShort4 = 9
    hipResViewFormatSignedShort1 = 10
    hipResViewFormatSignedShort2 = 11
    hipResViewFormatSignedShort4 = 12
    hipResViewFormatUnsignedInt1 = 13
    hipResViewFormatUnsignedInt2 = 14
    hipResViewFormatUnsignedInt4 = 15
    hipResViewFormatSignedInt1 = 16
    hipResViewFormatSignedInt2 = 17
    hipResViewFormatSignedInt4 = 18
    hipResViewFormatHalf1 = 19
    hipResViewFormatHalf2 = 20
    hipResViewFormatHalf4 = 21
    hipResViewFormatFloat1 = 22
    hipResViewFormatFloat2 = 23
    hipResViewFormatFloat4 = 24
    hipResViewFormatUnsignedBlockCompressed1 = 25
    hipResViewFormatUnsignedBlockCompressed2 = 26
    hipResViewFormatUnsignedBlockCompressed3 = 27
    hipResViewFormatUnsignedBlockCompressed4 = 28
    hipResViewFormatSignedBlockCompressed4 = 29
    hipResViewFormatUnsignedBlockCompressed5 = 30
    hipResViewFormatSignedBlockCompressed5 = 31
    hipResViewFormatUnsignedBlockCompressed6H = 32
    hipResViewFormatSignedBlockCompressed6H = 33
    hipResViewFormatUnsignedBlockCompressed7 = 34

class HIPresourceViewFormat_enum(enum.IntEnum):
    HIP_RES_VIEW_FORMAT_NONE = 0
    HIP_RES_VIEW_FORMAT_UINT_1X8 = 1
    HIP_RES_VIEW_FORMAT_UINT_2X8 = 2
    HIP_RES_VIEW_FORMAT_UINT_4X8 = 3
    HIP_RES_VIEW_FORMAT_SINT_1X8 = 4
    HIP_RES_VIEW_FORMAT_SINT_2X8 = 5
    HIP_RES_VIEW_FORMAT_SINT_4X8 = 6
    HIP_RES_VIEW_FORMAT_UINT_1X16 = 7
    HIP_RES_VIEW_FORMAT_UINT_2X16 = 8
    HIP_RES_VIEW_FORMAT_UINT_4X16 = 9
    HIP_RES_VIEW_FORMAT_SINT_1X16 = 10
    HIP_RES_VIEW_FORMAT_SINT_2X16 = 11
    HIP_RES_VIEW_FORMAT_SINT_4X16 = 12
    HIP_RES_VIEW_FORMAT_UINT_1X32 = 13
    HIP_RES_VIEW_FORMAT_UINT_2X32 = 14
    HIP_RES_VIEW_FORMAT_UINT_4X32 = 15
    HIP_RES_VIEW_FORMAT_SINT_1X32 = 16
    HIP_RES_VIEW_FORMAT_SINT_2X32 = 17
    HIP_RES_VIEW_FORMAT_SINT_4X32 = 18
    HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 19
    HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 20
    HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 21
    HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 22
    HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 23
    HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 24
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
    HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 29
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
    HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 31
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 33
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34

class HIPresourceViewFormat(enum.IntEnum):
    HIP_RES_VIEW_FORMAT_NONE = 0
    HIP_RES_VIEW_FORMAT_UINT_1X8 = 1
    HIP_RES_VIEW_FORMAT_UINT_2X8 = 2
    HIP_RES_VIEW_FORMAT_UINT_4X8 = 3
    HIP_RES_VIEW_FORMAT_SINT_1X8 = 4
    HIP_RES_VIEW_FORMAT_SINT_2X8 = 5
    HIP_RES_VIEW_FORMAT_SINT_4X8 = 6
    HIP_RES_VIEW_FORMAT_UINT_1X16 = 7
    HIP_RES_VIEW_FORMAT_UINT_2X16 = 8
    HIP_RES_VIEW_FORMAT_UINT_4X16 = 9
    HIP_RES_VIEW_FORMAT_SINT_1X16 = 10
    HIP_RES_VIEW_FORMAT_SINT_2X16 = 11
    HIP_RES_VIEW_FORMAT_SINT_4X16 = 12
    HIP_RES_VIEW_FORMAT_UINT_1X32 = 13
    HIP_RES_VIEW_FORMAT_UINT_2X32 = 14
    HIP_RES_VIEW_FORMAT_UINT_4X32 = 15
    HIP_RES_VIEW_FORMAT_SINT_1X32 = 16
    HIP_RES_VIEW_FORMAT_SINT_2X32 = 17
    HIP_RES_VIEW_FORMAT_SINT_4X32 = 18
    HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 19
    HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 20
    HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 21
    HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 22
    HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 23
    HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 24
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
    HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 29
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
    HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 31
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 33
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34

class hipMemcpyKind(enum.IntEnum):
    hipMemcpyHostToHost = 0
    hipMemcpyHostToDevice = 1
    hipMemcpyDeviceToHost = 2
    hipMemcpyDeviceToDevice = 3
    hipMemcpyDefault = 4

class hipFunction_attribute(enum.IntEnum):
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
    HIP_FUNC_ATTRIBUTE_NUM_REGS = 4
    HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6
    HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    HIP_FUNC_ATTRIBUTE_MAX = 10

class hipPointer_attribute(enum.IntEnum):
    HIP_POINTER_ATTRIBUTE_CONTEXT = 1
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
    HIP_POINTER_ATTRIBUTE_HOST_POINTER = 4
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS = 5
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
    HIP_POINTER_ATTRIBUTE_BUFFER_ID = 7
    HIP_POINTER_ATTRIBUTE_IS_MANAGED = 8
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 10
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE = 12
    HIP_POINTER_ATTRIBUTE_MAPPED = 13
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17

def hipCreateChannelDesc(x,y,z,w,f) nogil:
    """
    """
    pass

class hipTextureAddressMode(enum.IntEnum):
    hipAddressModeWrap = 0
    hipAddressModeClamp = 1
    hipAddressModeMirror = 2
    hipAddressModeBorder = 3

class hipTextureFilterMode(enum.IntEnum):
    hipFilterModePoint = 0
    hipFilterModeLinear = 1

class hipTextureReadMode(enum.IntEnum):
    hipReadModeElementType = 0
    hipReadModeNormalizedFloat = 1

class hipSurfaceBoundaryMode(enum.IntEnum):
    hipBoundaryModeZero = 0
    hipBoundaryModeTrap = 1
    hipBoundaryModeClamp = 2

class hipDeviceP2PAttr(enum.IntEnum):
    hipDevP2PAttrPerformanceRank = 0
    hipDevP2PAttrAccessSupported = 1
    hipDevP2PAttrNativeAtomicSupported = 2
    hipDevP2PAttrHipArrayAccessSupported = 3

class hipLimit_t(enum.IntEnum):
    hipLimitStackSize = 0
    hipLimitPrintfFifoSize = 1
    hipLimitMallocHeapSize = 2
    hipLimitRange = 3

class hipMemoryAdvise(enum.IntEnum):
    hipMemAdviseSetReadMostly = 1
    hipMemAdviseUnsetReadMostly = 2
    hipMemAdviseSetPreferredLocation = 3
    hipMemAdviseUnsetPreferredLocation = 4
    hipMemAdviseSetAccessedBy = 5
    hipMemAdviseUnsetAccessedBy = 6
    hipMemAdviseSetCoarseGrain = 100
    hipMemAdviseUnsetCoarseGrain = 101

class hipMemRangeCoherencyMode(enum.IntEnum):
    hipMemRangeCoherencyModeFineGrain = 0
    hipMemRangeCoherencyModeCoarseGrain = 1
    hipMemRangeCoherencyModeIndeterminate = 2

class hipMemRangeAttribute(enum.IntEnum):
    hipMemRangeAttributeReadMostly = 1
    hipMemRangeAttributePreferredLocation = 2
    hipMemRangeAttributeAccessedBy = 3
    hipMemRangeAttributeLastPrefetchLocation = 4
    hipMemRangeAttributeCoherencyMode = 100

class hipMemPoolAttr(enum.IntEnum):
    hipMemPoolReuseFollowEventDependencies = 1
    hipMemPoolReuseAllowOpportunistic = 2
    hipMemPoolReuseAllowInternalDependencies = 3
    hipMemPoolAttrReleaseThreshold = 4
    hipMemPoolAttrReservedMemCurrent = 5
    hipMemPoolAttrReservedMemHigh = 6
    hipMemPoolAttrUsedMemCurrent = 7
    hipMemPoolAttrUsedMemHigh = 8

class hipMemLocationType(enum.IntEnum):
    hipMemLocationTypeInvalid = 0
    hipMemLocationTypeDevice = 1

class hipMemAccessFlags(enum.IntEnum):
    hipMemAccessFlagsProtNone = 0
    hipMemAccessFlagsProtRead = 1
    hipMemAccessFlagsProtReadWrite = 3

class hipMemAllocationType(enum.IntEnum):
    hipMemAllocationTypeInvalid = 0
    hipMemAllocationTypePinned = 1
    hipMemAllocationTypeMax = 2147483647

class hipMemAllocationHandleType(enum.IntEnum):
    hipMemHandleTypeNone = 0
    hipMemHandleTypePosixFileDescriptor = 1
    hipMemHandleTypeWin32 = 2
    hipMemHandleTypeWin32Kmt = 4

class hipJitOption(enum.IntEnum):
    hipJitOptionMaxRegisters = 0
    hipJitOptionThreadsPerBlock = 1
    hipJitOptionWallTime = 2
    hipJitOptionInfoLogBuffer = 3
    hipJitOptionInfoLogBufferSizeBytes = 4
    hipJitOptionErrorLogBuffer = 5
    hipJitOptionErrorLogBufferSizeBytes = 6
    hipJitOptionOptimizationLevel = 7
    hipJitOptionTargetFromContext = 8
    hipJitOptionTarget = 9
    hipJitOptionFallbackStrategy = 10
    hipJitOptionGenerateDebugInfo = 11
    hipJitOptionLogVerbose = 12
    hipJitOptionGenerateLineInfo = 13
    hipJitOptionCacheMode = 14
    hipJitOptionSm3xOpt = 15
    hipJitOptionFastCompile = 16
    hipJitOptionNumOptions = 17

class hipFuncAttribute(enum.IntEnum):
    hipFuncAttributeMaxDynamicSharedMemorySize = 8
    hipFuncAttributePreferredSharedMemoryCarveout = 9
    hipFuncAttributeMax = 10

class hipFuncCache_t(enum.IntEnum):
    hipFuncCachePreferNone = 0
    hipFuncCachePreferShared = 1
    hipFuncCachePreferL1 = 2
    hipFuncCachePreferEqual = 3

class hipSharedMemConfig(enum.IntEnum):
    hipSharedMemBankSizeDefault = 0
    hipSharedMemBankSizeFourByte = 1
    hipSharedMemBankSizeEightByte = 2

class hipExternalMemoryHandleType_enum(enum.IntEnum):
    hipExternalMemoryHandleTypeOpaqueFd = 1
    hipExternalMemoryHandleTypeOpaqueWin32 = 2
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3
    hipExternalMemoryHandleTypeD3D12Heap = 4
    hipExternalMemoryHandleTypeD3D12Resource = 5
    hipExternalMemoryHandleTypeD3D11Resource = 6
    hipExternalMemoryHandleTypeD3D11ResourceKmt = 7

class hipExternalMemoryHandleType(enum.IntEnum):
    hipExternalMemoryHandleTypeOpaqueFd = 1
    hipExternalMemoryHandleTypeOpaqueWin32 = 2
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3
    hipExternalMemoryHandleTypeD3D12Heap = 4
    hipExternalMemoryHandleTypeD3D12Resource = 5
    hipExternalMemoryHandleTypeD3D11Resource = 6
    hipExternalMemoryHandleTypeD3D11ResourceKmt = 7

class hipExternalSemaphoreHandleType_enum(enum.IntEnum):
    hipExternalSemaphoreHandleTypeOpaqueFd = 1
    hipExternalSemaphoreHandleTypeOpaqueWin32 = 2
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3
    hipExternalSemaphoreHandleTypeD3D12Fence = 4

class hipExternalSemaphoreHandleType(enum.IntEnum):
    hipExternalSemaphoreHandleTypeOpaqueFd = 1
    hipExternalSemaphoreHandleTypeOpaqueWin32 = 2
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3
    hipExternalSemaphoreHandleTypeD3D12Fence = 4

class hipGLDeviceList(enum.IntEnum):
    hipGLDeviceListAll = 1
    hipGLDeviceListCurrentFrame = 2
    hipGLDeviceListNextFrame = 3

class hipGraphicsRegisterFlags(enum.IntEnum):
    hipGraphicsRegisterFlagsNone = 0
    hipGraphicsRegisterFlagsReadOnly = 1
    hipGraphicsRegisterFlagsWriteDiscard = 2
    hipGraphicsRegisterFlagsSurfaceLoadStore = 4
    hipGraphicsRegisterFlagsTextureGather = 8

class hipGraphNodeType(enum.IntEnum):
    hipGraphNodeTypeKernel = 0
    hipGraphNodeTypeMemcpy = 1
    hipGraphNodeTypeMemset = 2
    hipGraphNodeTypeHost = 3
    hipGraphNodeTypeGraph = 4
    hipGraphNodeTypeEmpty = 5
    hipGraphNodeTypeWaitEvent = 6
    hipGraphNodeTypeEventRecord = 7
    hipGraphNodeTypeExtSemaphoreSignal = 8
    hipGraphNodeTypeExtSemaphoreWait = 9
    hipGraphNodeTypeMemcpyFromSymbol = 10
    hipGraphNodeTypeMemcpyToSymbol = 11
    hipGraphNodeTypeCount = 12

class hipKernelNodeAttrID(enum.IntEnum):
    hipKernelNodeAttributeAccessPolicyWindow = 1
    hipKernelNodeAttributeCooperative = 2

class hipAccessProperty(enum.IntEnum):
    hipAccessPropertyNormal = 0
    hipAccessPropertyStreaming = 1
    hipAccessPropertyPersisting = 2

class hipGraphExecUpdateResult(enum.IntEnum):
    hipGraphExecUpdateSuccess = 0
    hipGraphExecUpdateError = 1
    hipGraphExecUpdateErrorTopologyChanged = 2
    hipGraphExecUpdateErrorNodeTypeChanged = 3
    hipGraphExecUpdateErrorFunctionChanged = 4
    hipGraphExecUpdateErrorParametersChanged = 5
    hipGraphExecUpdateErrorNotSupported = 6
    hipGraphExecUpdateErrorUnsupportedFunctionChange = 7

class hipStreamCaptureMode(enum.IntEnum):
    hipStreamCaptureModeGlobal = 0
    hipStreamCaptureModeThreadLocal = 1
    hipStreamCaptureModeRelaxed = 2

class hipStreamCaptureStatus(enum.IntEnum):
    hipStreamCaptureStatusNone = 0
    hipStreamCaptureStatusActive = 1
    hipStreamCaptureStatusInvalidated = 2

class hipStreamUpdateCaptureDependenciesFlags(enum.IntEnum):
    hipStreamAddCaptureDependencies = 0
    hipStreamSetCaptureDependencies = 1

class hipGraphMemAttributeType(enum.IntEnum):
    hipGraphMemAttrUsedMemCurrent = 0
    hipGraphMemAttrUsedMemHigh = 1
    hipGraphMemAttrReservedMemCurrent = 2
    hipGraphMemAttrReservedMemHigh = 3

class hipUserObjectFlags(enum.IntEnum):
    hipUserObjectNoDestructorSync = 1

class hipUserObjectRetainFlags(enum.IntEnum):
    hipGraphUserObjectMove = 1

class hipGraphInstantiateFlags(enum.IntEnum):
    hipGraphInstantiateFlagAutoFreeOnLaunch = 1

class hipMemAllocationGranularity_flags(enum.IntEnum):
    hipMemAllocationGranularityMinimum = 0
    hipMemAllocationGranularityRecommended = 1

class hipMemHandleType(enum.IntEnum):
    hipMemHandleTypeGeneric = 0

class hipMemOperationType(enum.IntEnum):
    hipMemOperationTypeMap = 1
    hipMemOperationTypeUnmap = 2

class hipArraySparseSubresourceType(enum.IntEnum):
    hipArraySparseSubresourceTypeSparseLevel = 0
    hipArraySparseSubresourceTypeMiptail = 1

def hipInit(flags) nogil:
    """@defgroup API HIP API
    @{
    Defines the HIP API.  See the individual sections for more information.
    @defgroup Driver Initialization and Version
    @{
    This section describes the initializtion and version functions of HIP runtime API.
    @brief Explicitly initializes the HIP runtime.
    Most HIP APIs implicitly initialize the HIP runtime.
    This API provides control over the timing of the initialization.
    """
    pass

def hipDriverGetVersion(driverVersion) nogil:
    """@brief Returns the approximate HIP driver version.
    @param [out] driverVersion
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning The HIP feature set does not correspond to an exact CUDA SDK driver revision.
    This function always set *driverVersion to 4 as an approximation though HIP supports
    some features which were introduced in later CUDA SDK revisions.
    HIP apps code should not rely on the driver revision number here and should
    use arch feature flags to test device capabilities or conditional compilation.
    @see hipRuntimeGetVersion
    """
    pass

def hipRuntimeGetVersion(runtimeVersion) nogil:
    """@brief Returns the approximate HIP Runtime version.
    @param [out] runtimeVersion
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning The version definition of HIP runtime is different from CUDA.
    On AMD platform, the function returns HIP runtime version,
    while on NVIDIA platform, it returns CUDA runtime version.
    And there is no mapping/correlation between HIP version and CUDA version.
    @see hipDriverGetVersion
    """
    pass

def hipDeviceGet(device,ordinal) nogil:
    """@brief Returns a handle to a compute device
    @param [out] device
    @param [in] ordinal
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceComputeCapability(major,minor,device) nogil:
    """@brief Returns the compute capability of the device
    @param [out] major
    @param [out] minor
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceGetName(name,len,device) nogil:
    """@brief Returns an identifer string for the device.
    @param [out] name
    @param [in] len
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceGetUuid(uuid,device) nogil:
    """@brief Returns an UUID for the device.[BETA]
    @param [out] uuid
    @param [in] device
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotInitialized,
    #hipErrorDeinitialized
    """
    pass

def hipDeviceGetP2PAttribute(value,attr,srcDevice,dstDevice) nogil:
    """@brief Returns a value for attr of link between two devices
    @param [out] value
    @param [in] attr
    @param [in] srcDevice
    @param [in] dstDevice
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceGetPCIBusId(pciBusId,len,device) nogil:
    """@brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
    @param [out] pciBusId
    @param [in] len
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceGetByPCIBusId(device,pciBusId) nogil:
    """@brief Returns a handle to a compute device.
    @param [out] device handle
    @param [in] PCI Bus ID
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    """
    pass

def hipDeviceTotalMem(bytes,device) nogil:
    """@brief Returns the total amount of memory on the device.
    @param [out] bytes
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceSynchronize() nogil:
    """@}
    @defgroup Device Device Management
    @{
    This section describes the device management functions of HIP runtime API.
    @brief Waits on all active streams on current device
    When this command is invoked, the host thread gets blocked until all the commands associated
    with streams associated with the device. HIP does not support multiple blocking modes (yet!).
    @returns #hipSuccess
    @see hipSetDevice, hipDeviceReset
    """
    pass

def hipDeviceReset() nogil:
    """@brief The state of current device is discarded and updated to a fresh state.
    Calling this function deletes all streams created, memory allocated, kernels running, events
    created. Make sure that no other thread is using the device or streams, memory, kernels, events
    associated with the current device.
    @returns #hipSuccess
    @see hipDeviceSynchronize
    """
    pass

def hipSetDevice(deviceId) nogil:
    """@brief Set default device to be used for subsequent hip API calls from this thread.
    @param[in] deviceId Valid device in range 0...hipGetDeviceCount().
    Sets @p device as the default device for the calling host thread.  Valid device id's are 0...
    (hipGetDeviceCount()-1).
    Many HIP APIs implicitly use the "default device" :
    - Any device memory subsequently allocated from this host thread (using hipMalloc) will be
    allocated on device.
    - Any streams or events created from this host thread will be associated with device.
    - Any kernels launched from this host thread (using hipLaunchKernel) will be executed on device
    (unless a specific stream is specified, in which case the device associated with that stream will
    be used).
    This function may be called from any host thread.  Multiple host threads may use the same device.
    This function does no synchronization with the previous or new device, and has very little
    runtime overhead. Applications can use hipSetDevice to quickly switch the default device before
    making a HIP runtime call which uses the default device.
    The default device is stored in thread-local-storage for each thread.
    Thread-pool implementations may inherit the default device of the previous thread.  A good
    practice is to always call hipSetDevice at the start of HIP coding sequency to establish a known
    standard device.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorDeviceAlreadyInUse
    @see hipGetDevice, hipGetDeviceCount
    """
    pass

def hipGetDevice(deviceId) nogil:
    """@brief Return the default device id for the calling host thread.
    @param [out] device *device is written with the default device
    HIP maintains an default device for each thread using thread-local-storage.
    This device is used implicitly for HIP runtime APIs called by this thread.
    hipGetDevice returns in * @p device the default device for the calling host thread.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see hipSetDevice, hipGetDevicesizeBytes
    """
    pass

def hipGetDeviceCount(count) nogil:
    """@brief Return number of compute-capable devices.
    @param [output] count Returns number of compute-capable devices.
    @returns #hipSuccess, #hipErrorNoDevice
    Returns in @p *count the number of devices that have ability to run compute commands.  If there
    are no such devices, then @ref hipGetDeviceCount will return #hipErrorNoDevice. If 1 or more
    devices can be found, then hipGetDeviceCount returns #hipSuccess.
    """
    pass

def hipDeviceGetAttribute(pi,attr,deviceId) nogil:
    """@brief Query for a specific device attribute.
    @param [out] pi pointer to value to return
    @param [in] attr attribute to query
    @param [in] deviceId which device to query for information
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    """
    pass

def hipDeviceGetDefaultMemPool(mem_pool,device) nogil:
    """@brief Returns the default memory pool of the specified device
    @param [out] mem_pool Default memory pool to return
    @param [in] device    Device index for query the default memory pool
    @returns #chipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotSupported
    @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceSetMemPool(device,mem_pool) nogil:
    """@brief Sets the current memory pool of a device
    The memory pool must be local to the specified device.
    @p hipMallocAsync allocates from the current mempool of the provided stream's device.
    By default, a device's current memory pool is its default memory pool.
    @note Use @p hipMallocFromPoolAsync for asynchronous memory allocations from a device
    different than the one the stream runs on.
    @param [in] device   Device index for the update
    @param [in] mem_pool Memory pool for update as the current on the specified device
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice, #hipErrorNotSupported
    @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceGetMemPool(mem_pool,device) nogil:
    """@brief Gets the current memory pool for the specified device
    Returns the last pool provided to @p hipDeviceSetMemPool for this device
    or the device's default memory pool if @p hipDeviceSetMemPool has never been called.
    By default the current mempool is the default mempool for a device,
    otherwise the returned pool must have been set with @p hipDeviceSetMemPool.
    @param [out] mem_pool Current memory pool on the specified device
    @param [in] device    Device index to query the current memory pool
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGetDeviceProperties(prop,deviceId) nogil:
    """@brief Returns device properties.
    @param [out] prop written with device properties
    @param [in]  deviceId which device to query for information
    @return #hipSuccess, #hipErrorInvalidDevice
    @bug HCC always returns 0 for maxThreadsPerMultiProcessor
    @bug HCC always returns 0 for regsPerBlock
    @bug HCC always returns 0 for l2CacheSize
    Populates hipGetDeviceProperties with information for the specified device.
    """
    pass

def hipDeviceSetCacheConfig(cacheConfig) nogil:
    """@brief Set L1/Shared cache partition.
    @param [in] cacheConfig
    @returns #hipSuccess, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    on those architectures.
    """
    pass

def hipDeviceGetCacheConfig(cacheConfig) nogil:
    """@brief Get Cache configuration for a specific Device
    @param [out] cacheConfig
    @returns #hipSuccess, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    on those architectures.
    """
    pass

def hipDeviceGetLimit(pValue,limit) nogil:
    """@brief Get Resource limits of current device
    @param [out] pValue
    @param [in]  limit
    @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
    Note: Currently, only hipLimitMallocHeapSize is available
    """
    pass

def hipDeviceSetLimit(limit,value) nogil:
    """@brief Set Resource limits of current device
    @param [in] limit
    @param [in] value
    @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
    """
    pass

def hipDeviceGetSharedMemConfig(pConfig) nogil:
    """@brief Returns bank width of shared memory for current device
    @param [out] pConfig
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipGetDeviceFlags(flags) nogil:
    """@brief Gets the flags set for current device
    @param [out] flags
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    """
    pass

def hipDeviceSetSharedMemConfig(config) nogil:
    """@brief The bank width of shared memory on current device is set
    @param [in] config
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipSetDeviceFlags(flags) nogil:
    """@brief The current device behavior is changed according the flags passed.
    @param [in] flags
    The schedule flags impact how HIP waits for the completion of a command running on a device.
    hipDeviceScheduleSpin         : HIP runtime will actively spin in the thread which submitted the
    work until the command completes.  This offers the lowest latency, but will consume a CPU core
    and may increase power. hipDeviceScheduleYield        : The HIP runtime will yield the CPU to
    system so that other tasks can use it.  This may increase latency to detect the completion but
    will consume less power and is friendlier to other tasks in the system.
    hipDeviceScheduleBlockingSync : On ROCm platform, this is a synonym for hipDeviceScheduleYield.
    hipDeviceScheduleAuto         : Use a hueristic to select between Spin and Yield modes.  If the
    number of HIP contexts is greater than the number of logical processors in the system, use Spin
    scheduling.  Else use Yield scheduling.
    hipDeviceMapHost              : Allow mapping host memory.  On ROCM, this is always allowed and
    the flag is ignored. hipDeviceLmemResizeToMax      : @warning ROCm silently ignores this flag.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorSetOnActiveProcess
    """
    pass

def hipChooseDevice(device,prop) nogil:
    """@brief Device which matches hipDeviceProp_t is returned
    @param [out] device ID
    @param [in]  device properties pointer
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipExtGetLinkTypeAndHopCount(device1,device2,linktype,hopcount) nogil:
    """@brief Returns the link type and hop count between two devices
    @param [in] device1 Ordinal for device1
    @param [in] device2 Ordinal for device2
    @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
    @param [out] hopcount Returns the hop count between the two devices
    Queries and returns the HSA link type and the hop count between the two specified devices.
    @returns #hipSuccess, #hipInvalidDevice, #hipErrorRuntimeOther
    """
    pass

def hipIpcGetMemHandle(handle,devPtr) nogil:
    """@brief Gets an interprocess memory handle for an existing device memory
    allocation
    Takes a pointer to the base of an existing device memory allocation created
    with hipMalloc and exports it for use in another process. This is a
    lightweight operation and may be called multiple times on an allocation
    without adverse effects.
    If a region of memory is freed with hipFree and a subsequent call
    to hipMalloc returns memory with the same device address,
    hipIpcGetMemHandle will return a unique handle for the
    new memory.
    @param handle - Pointer to user allocated hipIpcMemHandle to return
    the handle in.
    @param devPtr - Base pointer to previously allocated device memory
    @returns
    hipSuccess,
    hipErrorInvalidHandle,
    hipErrorOutOfMemory,
    hipErrorMapFailed,
    """
    pass

def hipIpcOpenMemHandle(devPtr,handle,flags) nogil:
    """@brief Opens an interprocess memory handle exported from another process
    and returns a device pointer usable in the local process.
    Maps memory exported from another process with hipIpcGetMemHandle into
    the current device address space. For contexts on different devices
    hipIpcOpenMemHandle can attempt to enable peer access between the
    devices as if the user called hipDeviceEnablePeerAccess. This behavior is
    controlled by the hipIpcMemLazyEnablePeerAccess flag.
    hipDeviceCanAccessPeer can determine if a mapping is possible.
    Contexts that may open hipIpcMemHandles are restricted in the following way.
    hipIpcMemHandles from each device in a given process may only be opened
    by one context per device per other process.
    Memory returned from hipIpcOpenMemHandle must be freed with
    hipIpcCloseMemHandle.
    Calling hipFree on an exported memory region before calling
    hipIpcCloseMemHandle in the importing context will result in undefined
    behavior.
    @param devPtr - Returned device pointer
    @param handle - hipIpcMemHandle to open
    @param flags  - Flags for this operation. Must be specified as hipIpcMemLazyEnablePeerAccess
    @returns
    hipSuccess,
    hipErrorMapFailed,
    hipErrorInvalidHandle,
    hipErrorTooManyPeers
    @note During multiple processes, using the same memory handle opened by the current context,
    there is no guarantee that the same device poiter will be returned in @p *devPtr.
    This is diffrent from CUDA.
    """
    pass

def hipIpcCloseMemHandle(devPtr) nogil:
    """@brief Close memory mapped with hipIpcOpenMemHandle
    Unmaps memory returnd by hipIpcOpenMemHandle. The original allocation
    in the exporting process as well as imported mappings in other processes
    will be unaffected.
    Any resources used to enable peer access will be freed if this is the
    last mapping using them.
    @param devPtr - Device pointer returned by hipIpcOpenMemHandle
    @returns
    hipSuccess,
    hipErrorMapFailed,
    hipErrorInvalidHandle,
    """
    pass

def hipIpcGetEventHandle(handle,event) nogil:
    """@brief Gets an opaque interprocess handle for an event.
    This opaque handle may be copied into other processes and opened with hipIpcOpenEventHandle.
    Then hipEventRecord, hipEventSynchronize, hipStreamWaitEvent and hipEventQuery may be used in
    either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
    will result in undefined behavior.
    @param[out]  handle Pointer to hipIpcEventHandle to return the opaque event handle
    @param[in]   event  Event allocated with hipEventInterprocess and hipEventDisableTiming flags
    @returns #hipSuccess, #hipErrorInvalidConfiguration, #hipErrorInvalidValue
    """
    pass

def hipIpcOpenEventHandle(event,handle) nogil:
    """@brief Opens an interprocess event handles.
    Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle. The returned
    hipEvent_t behaves like a locally created event with the hipEventDisableTiming flag specified. This event
    need be freed with hipEventDestroy. Operations on the imported event after the exported event has been freed
    with hipEventDestroy will result in undefined behavior. If the function is called within the same process where
    handle is returned by hipIpcGetEventHandle, it will return hipErrorInvalidContext.
    @param[out]  event  Pointer to hipEvent_t to return the event
    @param[in]   handle The opaque interprocess handle to open
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext
    """
    pass

def hipFuncSetAttribute(func,attr,value) nogil:
    """@}
    @defgroup Execution Execution Control
    @{
    This section describes the execution control functions of HIP runtime API.
    @brief Set attribute for a specific function
    @param [in] func;
    @param [in] attr;
    @param [in] value;
    @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipFuncSetCacheConfig(func,config) nogil:
    """@brief Set Cache configuration for a specific function
    @param [in] config;
    @returns #hipSuccess, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    on those architectures.
    """
    pass

def hipFuncSetSharedMemConfig(func,config) nogil:
    """@brief Set shared memory configuation for a specific function
    @param [in] func
    @param [in] config
    @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipGetLastError() nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Error Error Handling
    @{
    This section describes the error handling functions of HIP runtime API.
    @brief Return last error returned by any HIP runtime API call and resets the stored error code to
    #hipSuccess
    @returns return code from last HIP called from the active host thread
    Returns the last error that has been returned by any of the runtime calls in the same host
    thread, and then resets the saved error to #hipSuccess.
    @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipPeekAtLastError() nogil:
    """@brief Return last error returned by any HIP runtime API call.
    @return #hipSuccess
    Returns the last error that has been returned by any of the runtime calls in the same host
    thread. Unlike hipGetLastError, this function does not reset the saved error code.
    @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipGetErrorName(hip_error) nogil:
    """@brief Return hip error as text string form.
    @param hip_error Error code to convert to name.
    @return const char pointer to the NULL-terminated error name
    @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipGetErrorString(hipError) nogil:
    """@brief Return handy text string message to explain the error which occurred
    @param hipError Error code to convert to string.
    @return const char pointer to the NULL-terminated error string
    @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipDrvGetErrorName(hipError,errorString) nogil:
    """@brief Return hip error as text string form.
    @param [in] hipError Error code to convert to string.
    @param [out] const char pointer to the NULL-terminated error string
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipDrvGetErrorString(hipError,errorString) nogil:
    """@brief Return handy text string message to explain the error which occurred
    @param [in] hipError Error code to convert to string.
    @param [out] const char pointer to the NULL-terminated error string
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipStreamCreate(stream) nogil:
    """@brief Create an asynchronous stream.
    @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
    newly created stream.
    @return #hipSuccess, #hipErrorInvalidValue
    Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
    reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
    the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
    used by the stream, applicaiton must call hipStreamDestroy.
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipStreamCreateWithFlags(stream,flags) nogil:
    """@brief Create an asynchronous stream.
    @param[in, out] stream Pointer to new stream
    @param[in ] flags to control stream creation.
    @return #hipSuccess, #hipErrorInvalidValue
    Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
    reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
    the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
    used by the stream, applicaiton must call hipStreamDestroy. Flags controls behavior of the
    stream.  See #hipStreamDefault, #hipStreamNonBlocking.
    @see hipStreamCreate, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipStreamCreateWithPriority(stream,flags,priority) nogil:
    """@brief Create an asynchronous stream with the specified priority.
    @param[in, out] stream Pointer to new stream
    @param[in ] flags to control stream creation.
    @param[in ] priority of the stream. Lower numbers represent higher priorities.
    @return #hipSuccess, #hipErrorInvalidValue
    Create a new asynchronous stream with the specified priority.  @p stream returns an opaque handle
    that can be used to reference the newly created stream in subsequent hipStream* commands.  The
    stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
    To release the memory used by the stream, applicaiton must call hipStreamDestroy. Flags controls
    behavior of the stream.  See #hipStreamDefault, #hipStreamNonBlocking.
    @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipDeviceGetStreamPriorityRange(leastPriority,greatestPriority) nogil:
    """@brief Returns numerical values that correspond to the least and greatest stream priority.
    @param[in, out] leastPriority pointer in which value corresponding to least priority is returned.
    @param[in, out] greatestPriority pointer in which value corresponding to greatest priority is returned.
    Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least
    and greatest stream priority respectively. Stream priorities follow a convention where lower numbers
    imply greater priorities. The range of meaningful stream priorities is given by
    [*greatestPriority, *leastPriority]. If the user attempts to create a stream with a priority value
    that is outside the the meaningful range as specified by this API, the priority is automatically
    clamped to within the valid range.
    """
    pass

def hipStreamDestroy(stream) nogil:
    """@brief Destroys the specified stream.
    @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
    newly created stream.
    @return #hipSuccess #hipErrorInvalidHandle
    Destroys the specified stream.
    If commands are still executing on the specified stream, some may complete execution before the
    queue is deleted.
    The queue may be destroyed while some commands are still inflight, or may wait for all commands
    queued to the stream before destroying it.
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamQuery, hipStreamWaitEvent,
    hipStreamSynchronize
    """
    pass

def hipStreamQuery(stream) nogil:
    """@brief Return #hipSuccess if all of the operations in the specified @p stream have completed, or
    #hipErrorNotReady if not.
    @param[in] stream stream to query
    @return #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle
    This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
    host threads are sending work to the stream, the status may change immediately after the function
    is called.  It is typically used for debug.
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamSynchronize,
    hipStreamDestroy
    """
    pass

def hipStreamSynchronize(stream) nogil:
    """@brief Wait for all commands in stream to complete.
    @param[in] stream stream identifier.
    @return #hipSuccess, #hipErrorInvalidHandle
    This command is host-synchronous : the host will block until the specified stream is empty.
    This command follows standard null-stream semantics.  Specifically, specifying the null stream
    will cause the command to wait for other streams on the same device to complete all pending
    operations.
    This command honors the hipDeviceLaunchBlocking flag, which controls whether the wait is active
    or blocking.
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipStreamWaitEvent(stream,event,flags) nogil:
    """@brief Make the specified compute stream wait for an event
    @param[in] stream stream to make wait.
    @param[in] event event to wait on
    @param[in] flags control operation [must be 0]
    @return #hipSuccess, #hipErrorInvalidHandle
    This function inserts a wait operation into the specified stream.
    All future work submitted to @p stream will wait until @p event reports completion before
    beginning execution.
    This function only waits for commands in the current stream to complete.  Notably,, this function
    does not impliciy wait for commands in the default stream to complete, even if the specified
    stream is created with hipStreamNonBlocking = 0.
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamDestroy
    """
    pass

def hipStreamGetFlags(stream,flags) nogil:
    """@brief Return flags associated with this stream.
    @param[in] stream stream to be queried
    @param[in,out] flags Pointer to an unsigned integer in which the stream's flags are returned
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
    @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
    Return flags associated with this stream in *@p flags.
    @see hipStreamCreateWithFlags
    """
    pass

def hipStreamGetPriority(stream,priority) nogil:
    """@brief Query the priority of a stream.
    @param[in] stream stream to be queried
    @param[in,out] priority Pointer to an unsigned integer in which the stream's priority is returned
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
    @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
    Query the priority of a stream. The priority is returned in in priority.
    @see hipStreamCreateWithFlags
    """
    pass

def hipExtStreamCreateWithCUMask(stream,cuMaskSize,cuMask) nogil:
    """@brief Create an asynchronous stream with the specified CU mask.
    @param[in, out] stream Pointer to new stream
    @param[in ] cuMaskSize Size of CU mask bit array passed in.
    @param[in ] cuMask Bit-vector representing the CU mask. Each active bit represents using one CU.
    The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
    CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are ignored.
    It is user's responsibility to make sure the input is meaningful.
    @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
    Create a new asynchronous stream with the specified CU mask.  @p stream returns an opaque handle
    that can be used to reference the newly created stream in subsequent hipStream* commands.  The
    stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
    To release the memory used by the stream, application must call hipStreamDestroy.
    @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipExtStreamGetCUMask(stream,cuMaskSize,cuMask) nogil:
    """@brief Get CU mask associated with an asynchronous stream
    @param[in] stream stream to be queried
    @param[in] cuMaskSize number of the block of memories (uint32_t *) allocated by user
    @param[out] cuMask Pointer to a pre-allocated block of memories (uint32_t *) in which
    the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
    each active bit represents one active CU
    @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
    @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipStreamAddCallback(stream,callback,userData,flags) nogil:
    """@brief Adds a callback to be called on the host after all currently enqueued
    items in the stream have completed.  For each
    hipStreamAddCallback call, a callback will be executed exactly once.
    The callback will block later work in the stream until it is finished.
    @param[in] stream   - Stream to add callback to
    @param[in] callback - The function to call once preceding stream operations are complete
    @param[in] userData - User specified data to be passed to the callback function
    @param[in] flags    - Reserved for future use, must be 0
    @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorNotSupported
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamSynchronize,
    hipStreamWaitEvent, hipStreamDestroy, hipStreamCreateWithPriority
    """
    pass

def hipStreamWaitValue32(stream,ptr,value,flags,mask) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup StreamM Stream Memory Operations
    @{
    This section describes Stream Memory Wait and Write functions of HIP runtime API.
    @brief Enqueues a wait command to the stream.[BETA]
    @param [in] stream - Stream identifier
    @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
    @param [in] value  - Value to be used in compare operation
    @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
    hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor
    @param [in] mask   - Mask to be applied on value at memory before it is compared with value,
    default value is set to enable every bit
    @returns #hipSuccess, #hipErrorInvalidValue
    Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
    not execute until the defined wait condition is true.
    hipStreamWaitValueGte: waits until *ptr&mask >= value
    hipStreamWaitValueEq : waits until *ptr&mask == value
    hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
    hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
    @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
    @note Support for hipStreamWaitValue32 can be queried using 'hipDeviceGetAttribute()' and
    'hipDeviceAttributeCanUseStreamWaitValue' flag.
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue64, hipStreamWriteValue64,
    hipStreamWriteValue32, hipDeviceGetAttribute
    """
    pass

def hipStreamWaitValue64(stream,ptr,value,flags,mask) nogil:
    """@brief Enqueues a wait command to the stream.[BETA]
    @param [in] stream - Stream identifier
    @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
    @param [in] value  - Value to be used in compare operation
    @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
    hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor.
    @param [in] mask   - Mask to be applied on value at memory before it is compared with value
    default value is set to enable every bit
    @returns #hipSuccess, #hipErrorInvalidValue
    Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
    not execute until the defined wait condition is true.
    hipStreamWaitValueGte: waits until *ptr&mask >= value
    hipStreamWaitValueEq : waits until *ptr&mask == value
    hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
    hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
    @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
    @note Support for hipStreamWaitValue64 can be queried using 'hipDeviceGetAttribute()' and
    'hipDeviceAttributeCanUseStreamWaitValue' flag.
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue32, hipStreamWriteValue64,
    hipStreamWriteValue32, hipDeviceGetAttribute
    """
    pass

def hipStreamWriteValue32(stream,ptr,value,flags) nogil:
    """@brief Enqueues a write command to the stream.[BETA]
    @param [in] stream - Stream identifier
    @param [in] ptr    - Pointer to a GPU accessible memory object
    @param [in] value  - Value to be written
    @param [in] flags  - reserved, ignored for now, will be used in future releases
    @returns #hipSuccess, #hipErrorInvalidValue
    Enqueues a write command to the stream, write operation is performed after all earlier commands
    on this stream have completed the execution.
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
    hipStreamWaitValue64
    """
    pass

def hipStreamWriteValue64(stream,ptr,value,flags) nogil:
    """@brief Enqueues a write command to the stream.[BETA]
    @param [in] stream - Stream identifier
    @param [in] ptr    - Pointer to a GPU accessible memory object
    @param [in] value  - Value to be written
    @param [in] flags  - reserved, ignored for now, will be used in future releases
    @returns #hipSuccess, #hipErrorInvalidValue
    Enqueues a write command to the stream, write operation is performed after all earlier commands
    on this stream have completed the execution.
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
    hipStreamWaitValue64
    """
    pass

def hipEventCreateWithFlags(event,flags) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Event Event Management
    @{
    This section describes the event management functions of HIP runtime API.
    @brief Create an event with the specified flags
    @param[in,out] event Returns the newly created event.
    @param[in] flags     Flags to control event behavior.  Valid values are #hipEventDefault,
     #hipEventBlockingSync, #hipEventDisableTiming, #hipEventInterprocess
    #hipEventDefault : Default flag.  The event will use active synchronization and will support
     timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a
     CPU to poll on the event.
    #hipEventBlockingSync : The event will use blocking synchronization : if hipEventSynchronize is
     called on this event, the thread will block until the event completes.  This can increase latency
     for the synchroniation but can result in lower power and more resources for other CPU threads.
    #hipEventDisableTiming : Disable recording of timing information. Events created with this flag
     would not record profiling data and provide best performance if used for synchronization.
    #hipEventInterprocess : The event can be used as an interprocess event. hipEventDisableTiming
     flag also must be set when hipEventInterprocess flag is set.
    @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
     #hipErrorLaunchFailure, #hipErrorOutOfMemory
    @see hipEventCreate, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
    """
    pass

def hipEventCreate(event) nogil:
    """Create an event
    @param[in,out] event Returns the newly created event.
    @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
    #hipErrorLaunchFailure, #hipErrorOutOfMemory
    @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize,
    hipEventDestroy, hipEventElapsedTime
    """
    pass

def hipEventRecord(event,stream) nogil:
    """
    """
    pass

def hipEventDestroy(event) nogil:
    """@brief Destroy the specified event.
    @param[in] event Event to destroy.
    @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
    #hipErrorLaunchFailure
    Releases memory associated with the event.  If the event is recording but has not completed
    recording when hipEventDestroy() is called, the function will return immediately and the
    completion_future resources will be released later, when the hipDevice is synchronized.
    @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize, hipEventRecord,
    hipEventElapsedTime
    @returns #hipSuccess
    """
    pass

def hipEventSynchronize(event) nogil:
    """@brief Wait for an event to complete.
    This function will block until the event is ready, waiting for all previous work in the stream
    specified when event was recorded with hipEventRecord().
    If hipEventRecord() has not been called on @p event, this function returns immediately.
    TODO-hip- This function needs to support hipEventBlockingSync parameter.
    @param[in] event Event on which to wait.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized,
    #hipErrorInvalidHandle, #hipErrorLaunchFailure
    @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
    hipEventElapsedTime
    """
    pass

def hipEventElapsedTime(ms,start,stop) nogil:
    """@brief Return the elapsed time between two events.
    @param[out] ms : Return time between start and stop in ms.
    @param[in]   start : Start event.
    @param[in]   stop  : Stop event.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotReady, #hipErrorInvalidHandle,
    #hipErrorNotInitialized, #hipErrorLaunchFailure
    Computes the elapsed time between two events. Time is computed in ms, with
    a resolution of approximately 1 us.
    Events which are recorded in a NULL stream will block until all commands
    on all other streams complete execution, and then record the timestamp.
    Events which are recorded in a non-NULL stream will record their timestamp
    when they reach the head of the specified stream, after all previous
    commands in that stream have completed executing.  Thus the time that
    the event recorded may be significantly after the host calls hipEventRecord().
    If hipEventRecord() has not been called on either event, then #hipErrorInvalidHandle is
    returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
    recorded on one or both events (that is, hipEventQuery() would return #hipErrorNotReady on at
    least one of the events), then #hipErrorNotReady is returned.
    @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
    hipEventSynchronize
    """
    pass

def hipEventQuery(event) nogil:
    """@brief Query event status
    @param[in] event Event to query.
    @returns #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle, #hipErrorInvalidValue,
    #hipErrorNotInitialized, #hipErrorLaunchFailure
    Query the status of the specified event.  This function will return #hipSuccess if all
    commands in the appropriate stream (specified to hipEventRecord()) have completed.  If that work
    has not completed, or if hipEventRecord() was not called on the event, then #hipErrorNotReady is
    returned.
    @see hipEventCreate, hipEventCreateWithFlags, hipEventRecord, hipEventDestroy,
    hipEventSynchronize, hipEventElapsedTime
    """
    pass

def hipPointerGetAttributes(attributes,ptr) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Memory Memory Management
    @{
    This section describes the memory management functions of HIP runtime API.
    The following CUDA APIs are not currently supported:
    - cudaMalloc3D
    - cudaMalloc3DArray
    - TODO - more 2D, 3D, array APIs here.
    @brief Return attributes for the specified pointer
    @param [out]  attributes  attributes for the specified pointer
    @param [in]   ptr         pointer to get attributes for
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see hipPointerGetAttribute
    """
    pass

def hipPointerGetAttribute(data,attribute,ptr) nogil:
    """@brief Returns information about the specified pointer.[BETA]
    @param [in, out] data     returned pointer attribute value
    @param [in]      atribute attribute to query for
    @param [in]      ptr      pointer to get attributes for
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipPointerGetAttributes
    """
    pass

def hipDrvPointerGetAttributes(numAttributes,attributes,data,ptr) nogil:
    """@brief Returns information about the specified pointer.[BETA]
    @param [in]  numAttributes   number of attributes to query for
    @param [in]  attributes      attributes to query for
    @param [in, out] data        a two-dimensional containing pointers to memory locations
    where the result of each attribute query will be written to
    @param [in]  ptr             pointer to get attributes for
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipPointerGetAttribute
    """
    pass

def hipImportExternalSemaphore(extSem_out,semHandleDesc) nogil:
    """@brief Imports an external semaphore.
    @param[out] extSem_out  External semaphores to be waited on
    @param[in] semHandleDesc Semaphore import handle descriptor
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipSignalExternalSemaphoresAsync(extSemArray,paramsArray,numExtSems,stream) nogil:
    """@brief Signals a set of external semaphore objects.
    @param[in] extSem_out  External semaphores to be waited on
    @param[in] paramsArray Array of semaphore parameters
    @param[in] numExtSems Number of semaphores to wait on
    @param[in] stream Stream to enqueue the wait operations in
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipWaitExternalSemaphoresAsync(extSemArray,paramsArray,numExtSems,stream) nogil:
    """@brief Waits on a set of external semaphore objects
    @param[in] extSem_out  External semaphores to be waited on
    @param[in] paramsArray Array of semaphore parameters
    @param[in] numExtSems Number of semaphores to wait on
    @param[in] stream Stream to enqueue the wait operations in
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipDestroyExternalSemaphore(extSem) nogil:
    """@brief Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.
    @param[in] extSem handle to an external memory object
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipImportExternalMemory(extMem_out,memHandleDesc) nogil:
    """@brief Imports an external memory object.
    @param[out] extMem_out  Returned handle to an external memory object
    @param[in]  memHandleDesc Memory import handle descriptor
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipExternalMemoryGetMappedBuffer(devPtr,extMem,bufferDesc) nogil:
    """@brief Maps a buffer onto an imported memory object.
    @param[out] devPtr Returned device pointer to buffer
    @param[in]  extMem  Handle to external memory object
    @param[in]  bufferDesc  Buffer descriptor
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipDestroyExternalMemory(extMem) nogil:
    """@brief Destroys an external memory object.
    @param[in] extMem  External memory object to be destroyed
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipMalloc(ptr,size) nogil:
    """@brief Allocate memory on the default accelerator
    @param[out] ptr Pointer to the allocated memory
    @param[in]  size Requested memory size
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
    @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
    hipHostFree, hipHostMalloc
    """
    pass

def hipExtMallocWithFlags(ptr,sizeBytes,flags) nogil:
    """@brief Allocate memory on the default accelerator
    @param[out] ptr Pointer to the allocated memory
    @param[in]  size Requested memory size
    @param[in]  flags Type of memory allocation
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
    @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
    hipHostFree, hipHostMalloc
    """
    pass

def hipMallocHost(ptr,size) nogil:
    """@brief Allocate pinned host memory [Deprecated]
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @deprecated use hipHostMalloc() instead
    """
    pass

def hipMemAllocHost(ptr,size) nogil:
    """@brief Allocate pinned host memory [Deprecated]
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @deprecated use hipHostMalloc() instead
    """
    pass

def hipHostMalloc(ptr,size,flags) nogil:
    """@brief Allocate device accessible page locked host memory
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    @param[in]  flags Type of host memory allocation
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @see hipSetDeviceFlags, hipHostFree
    """
    pass

def hipMallocManaged(dev_ptr,size,flags) nogil:
    """-------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @addtogroup MemoryM Managed Memory
    @{
    @ingroup Memory
    This section describes the managed memory management functions of HIP runtime API.
    @brief Allocates memory that will be automatically managed by HIP.
    @param [out] dev_ptr - pointer to allocated device memory
    @param [in]  size    - requested allocation size in bytes
    @param [in]  flags   - must be either hipMemAttachGlobal or hipMemAttachHost
    (defaults to hipMemAttachGlobal)
    @returns #hipSuccess, #hipErrorMemoryAllocation, #hipErrorNotSupported, #hipErrorInvalidValue
    """
    pass

def hipMemPrefetchAsync(dev_ptr,count,device,stream) nogil:
    """@brief Prefetches memory to the specified destination device using HIP.
    @param [in] dev_ptr  pointer to be prefetched
    @param [in] count    size in bytes for prefetching
    @param [in] device   destination device to prefetch to
    @param [in] stream   stream to enqueue prefetch operation
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemAdvise(dev_ptr,count,advice,device) nogil:
    """@brief Advise about the usage of a given memory range to HIP.
    @param [in] dev_ptr  pointer to memory to set the advice for
    @param [in] count    size in bytes of the memory range
    @param [in] advice   advice to be applied for the specified memory range
    @param [in] device   device to apply the advice for
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemRangeGetAttribute(data,data_size,attribute,dev_ptr,count) nogil:
    """@brief Query an attribute of a given memory range in HIP.
    @param [in,out] data   a pointer to a memory location where the result of each
    attribute query will be written to
    @param [in] data_size  the size of data
    @param [in] attribute  the attribute to query
    @param [in] dev_ptr    start of the range to query
    @param [in] count      size of the range to query
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemRangeGetAttributes(data,data_sizes,attributes,num_attributes,dev_ptr,count) nogil:
    """@brief Query attributes of a given memory range in HIP.
    @param [in,out] data     a two-dimensional array containing pointers to memory locations
    where the result of each attribute query will be written to
    @param [in] data_sizes   an array, containing the sizes of each result
    @param [in] attributes   the attribute to query
    @param [in] num_attributes  an array of attributes to query (numAttributes and the number
    of attributes in this array should match)
    @param [in] dev_ptr      start of the range to query
    @param [in] count        size of the range to query
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipStreamAttachMemAsync(stream,dev_ptr,length,flags) nogil:
    """@brief Attach memory to a stream asynchronously in HIP.
    @param [in] stream     - stream in which to enqueue the attach operation
    @param [in] dev_ptr    - pointer to memory (must be a pointer to managed memory or
    to a valid host-accessible region of system-allocated memory)
    @param [in] length     - length of memory (defaults to zero)
    @param [in] flags      - must be one of hipMemAttachGlobal, hipMemAttachHost or
    hipMemAttachSingle (defaults to hipMemAttachSingle)
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMallocAsync(dev_ptr,size,stream) nogil:
    """@brief Allocates memory with stream ordered semantics
    Inserts a memory allocation operation into @p stream.
    A pointer to the allocated memory is returned immediately in *dptr.
    The allocation must not be accessed until the the allocation operation completes.
    The allocation comes from the memory pool associated with the stream's device.
    @note The default memory pool of a device contains device memory from that device.
    @note Basic stream ordering allows future work submitted into the same stream to use the allocation.
    Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
    operation completes before work submitted in a separate stream runs.
    @note During stream capture, this function results in the creation of an allocation node. In this case,
    the allocation is owned by the graph instead of the memory pool. The memory pool's properties
    are used to set the node's creation parameters.
    @param [out] dev_ptr  Returned device pointer of memory allocation
    @param [in] size      Number of bytes to allocate
    @param [in] stream    The stream establishing the stream ordering contract and
    the memory pool to allocate from
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
    @see hipMallocFromPoolAsync, hipFreeAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipFreeAsync(dev_ptr,stream) nogil:
    """@brief Frees memory with stream ordered semantics
    Inserts a free operation into @p stream.
    The allocation must not be used after stream execution reaches the free.
    After this API returns, accessing the memory from any subsequent work launched on the GPU
    or querying its pointer attributes results in undefined behavior.
    @note During stream capture, this function results in the creation of a free node and
    must therefore be passed the address of a graph allocation.
    @param [in] dev_ptr Pointer to device memory to free
    @param [in] stream  The stream, where the destruciton will occur according to the execution order
    @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported
    @see hipMallocFromPoolAsync, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolTrimTo(mem_pool,min_bytes_to_hold) nogil:
    """@brief Releases freed memory back to the OS
    Releases memory back to the OS until the pool contains fewer than @p min_bytes_to_keep
    reserved bytes, or there is no more memory that the allocator can safely release.
    The allocator cannot release OS allocations that back outstanding asynchronous allocations.
    The OS allocations may happen at different granularity from the user allocations.
    @note: Allocations that have not been freed count as outstanding.
    @note: Allocations that have been asynchronously freed but whose completion has
    not been observed on the host (eg. by a synchronize) can count as outstanding.
    @param[in] mem_pool          The memory pool to trim allocations
    @param[in] min_bytes_to_hold If the pool has less than min_bytes_to_hold reserved,
    then the TrimTo operation is a no-op.  Otherwise the memory pool will contain
    at least min_bytes_to_hold bytes reserved after the operation.
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolSetAttribute(mem_pool,attr,value) nogil:
    """@brief Sets attributes of a memory pool
    Supported attributes are:
    - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
    Amount of reserved memory in bytes to hold onto before trying
    to release memory back to the OS. When more than the release
    threshold bytes of memory are held by the memory pool, the
    allocator will try to release memory back to the OS on the
    next call to stream, event or context synchronize. (default 0)
    - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
    Allow @p hipMallocAsync to use memory asynchronously freed
    in another stream as long as a stream ordering dependency
    of the allocating stream on the free action exists.
    HIP events and null stream interactions can create the required
    stream ordered dependencies. (default enabled)
    - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
    Allow reuse of already completed frees when there is no dependency
    between the free and allocation. (default enabled)
    - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
    Allow @p hipMallocAsync to insert new stream dependencies
    in order to establish the stream ordering required to reuse
    a piece of memory released by @p hipFreeAsync (default enabled).
    @param [in] mem_pool The memory pool to modify
    @param [in] attr     The attribute to modify
    @param [in] value    Pointer to the value to assign
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolGetAttribute(mem_pool,attr,value) nogil:
    """@brief Gets attributes of a memory pool
    Supported attributes are:
    - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
    Amount of reserved memory in bytes to hold onto before trying
    to release memory back to the OS. When more than the release
    threshold bytes of memory are held by the memory pool, the
    allocator will try to release memory back to the OS on the
    next call to stream, event or context synchronize. (default 0)
    - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
    Allow @p hipMallocAsync to use memory asynchronously freed
    in another stream as long as a stream ordering dependency
    of the allocating stream on the free action exists.
    HIP events and null stream interactions can create the required
    stream ordered dependencies. (default enabled)
    - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
    Allow reuse of already completed frees when there is no dependency
    between the free and allocation. (default enabled)
    - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
    Allow @p hipMallocAsync to insert new stream dependencies
    in order to establish the stream ordering required to reuse
    a piece of memory released by @p hipFreeAsync (default enabled).
    @param [in] mem_pool The memory pool to get attributes of
    @param [in] attr     The attribute to get
    @param [in] value    Retrieved value
    @returns  #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolSetAccess(mem_pool,desc_list,count) nogil:
    """@brief Controls visibility of the specified pool between devices
    @param [in] mem_pool   Memory pool for acccess change
    @param [in] desc_list  Array of access descriptors. Each descriptor instructs the access to enable for a single gpu
    @param [in] count  Number of descriptors in the map array.
    @returns  #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolGetAccess(flags,mem_pool,location) nogil:
    """@brief Returns the accessibility of a pool from a device
    Returns the accessibility of the pool's memory from the specified location.
    @param [out] flags    Accessibility of the memory pool from the specified location/device
    @param [in] mem_pool   Memory pool being queried
    @param [in] location  Location/device for memory pool access
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolCreate(mem_pool,pool_props) nogil:
    """@brief Creates a memory pool
    Creates a HIP memory pool and returns the handle in @p mem_pool. The @p pool_props determines
    the properties of the pool such as the backing device and IPC capabilities.
    By default, the memory pool will be accessible from the device it is allocated on.
    @param [out] mem_pool    Contains createed memory pool
    @param [in] pool_props   Memory pool properties
    @note Specifying hipMemHandleTypeNone creates a memory pool that will not support IPC.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolDestroy,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolDestroy(mem_pool) nogil:
    """@brief Destroys the specified memory pool
    If any pointers obtained from this pool haven't been freed or
    the pool has free operations that haven't completed
    when @p hipMemPoolDestroy is invoked, the function will return immediately and the
    resources associated with the pool will be released automatically
    once there are no more outstanding allocations.
    Destroying the current mempool of a device sets the default mempool of
    that device as the current mempool for that device.
    @param [in] mem_pool Memory pool for destruction
    @note A device's default memory pool cannot be destroyed.
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMallocFromPoolAsync(dev_ptr,size,mem_pool,stream) nogil:
    """@brief Allocates memory from a specified pool with stream ordered semantics.
    Inserts an allocation operation into @p stream.
    A pointer to the allocated memory is returned immediately in @p dev_ptr.
    The allocation must not be accessed until the the allocation operation completes.
    The allocation comes from the specified memory pool.
    @note The specified memory pool may be from a device different than that of the specified @p stream.
    Basic stream ordering allows future work submitted into the same stream to use the allocation.
    Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
    operation completes before work submitted in a separate stream runs.
    @note During stream capture, this function results in the creation of an allocation node. In this case,
    the allocation is owned by the graph instead of the memory pool. The memory pool's properties
    are used to set the node's creation parameters.
    @param [out] dev_ptr Returned device pointer
    @param [in] size     Number of bytes to allocate
    @param [in] mem_pool The pool to allocate from
    @param [in] stream   The stream establishing the stream ordering semantic
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
    @see hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess,
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolExportToShareableHandle(shared_handle,mem_pool,handle_type,flags) nogil:
    """@brief Exports a memory pool to the requested handle type.
    Given an IPC capable mempool, create an OS handle to share the pool with another process.
    A recipient process can convert the shareable handle into a mempool with @p hipMemPoolImportFromShareableHandle.
    Individual pointers can then be shared with the @p hipMemPoolExportPointer and @p hipMemPoolImportPointer APIs.
    The implementation of what the shareable handle is and how it can be transferred is defined by the requested
    handle type.
    @note: To create an IPC capable mempool, create a mempool with a @p hipMemAllocationHandleType other
    than @p hipMemHandleTypeNone.
    @param [out] shared_handle Pointer to the location in which to store the requested handle
    @param [in] mem_pool       Pool to export
    @param [in] handle_type    The type of handle to create
    @param [in] flags          Must be 0
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    @see hipMemPoolImportFromShareableHandle
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolImportFromShareableHandle(mem_pool,shared_handle,handle_type,flags) nogil:
    """@brief Imports a memory pool from a shared handle.
    Specific allocations can be imported from the imported pool with @p hipMemPoolImportPointer.
    @note Imported memory pools do not support creating new allocations.
    As such imported memory pools may not be used in @p hipDeviceSetMemPool
    or @p hipMallocFromPoolAsync calls.
    @param [out] mem_pool     Returned memory pool
    @param [in] shared_handle OS handle of the pool to open
    @param [in] handle_type   The type of handle being imported
    @param [in] flags         Must be 0
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    @see hipMemPoolExportToShareableHandle
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolExportPointer(export_data,dev_ptr) nogil:
    """@brief Export data to share a memory pool allocation between processes.
    Constructs @p export_data for sharing a specific allocation from an already shared memory pool.
    The recipient process can import the allocation with the @p hipMemPoolImportPointer api.
    The data is not a handle and may be shared through any IPC mechanism.
    @param[out] export_data  Returned export data
    @param[in] dev_ptr       Pointer to memory being exported
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    @see hipMemPoolImportPointer
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolImportPointer(dev_ptr,mem_pool,export_data) nogil:
    """@brief Import a memory pool allocation from another process.
    Returns in @p dev_ptr a pointer to the imported memory.
    The imported memory must not be accessed before the allocation operation completes
    in the exporting process. The imported memory must be freed from all importing processes before
    being freed in the exporting process. The pointer may be freed with @p hipFree
    or @p hipFreeAsync. If @p hipFreeAsync is used, the free must be completed
    on the importing process before the free operation on the exporting process.
    @note The @p hipFreeAsync api may be used in the exporting process before
    the @p hipFreeAsync operation completes in its stream as long as the
    @p hipFreeAsync in the exporting process specifies a stream with
    a stream dependency on the importing process's @p hipFreeAsync.
    @param [out] dev_ptr     Pointer to imported memory
    @param [in] mem_pool     Memory pool from which to import a pointer
    @param [in] export_data  Data specifying the memory to import
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized, #hipErrorOutOfMemory
    @see hipMemPoolExportPointer
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipHostAlloc(ptr,size,flags) nogil:
    """@brief Allocate device accessible page locked host memory [Deprecated]
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    @param[in]  flags Type of host memory allocation
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @deprecated use hipHostMalloc() instead
    """
    pass

def hipHostGetDevicePointer(devPtr,hstPtr,flags) nogil:
    """@brief Get Device pointer from Host Pointer allocated through hipHostMalloc
    @param[out] dstPtr Device Pointer mapped to passed host pointer
    @param[in]  hstPtr Host Pointer allocated through hipHostMalloc
    @param[in]  flags Flags to be passed for extension
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    @see hipSetDeviceFlags, hipHostMalloc
    """
    pass

def hipHostGetFlags(flagsPtr,hostPtr) nogil:
    """@brief Return flags associated with host pointer
    @param[out] flagsPtr Memory location to store flags
    @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipHostMalloc
    """
    pass

def hipHostRegister(hostPtr,sizeBytes,flags) nogil:
    """@brief Register host memory so it can be accessed from the current device.
    @param[out] hostPtr Pointer to host memory to be registered.
    @param[in] sizeBytes size of the host memory
    @param[in] flags.  See below.
    Flags:
    - #hipHostRegisterDefault   Memory is Mapped and Portable
    - #hipHostRegisterPortable  Memory is considered registered by all contexts.  HIP only supports
    one context so this is always assumed true.
    - #hipHostRegisterMapped    Map the allocation into the address space for the current device.
    The device pointer can be obtained with #hipHostGetDevicePointer.
    After registering the memory, use #hipHostGetDevicePointer to obtain the mapped device pointer.
    On many systems, the mapped device pointer will have a different value than the mapped host
    pointer.  Applications must use the device pointer in device code, and the host pointer in device
    code.
    On some systems, registered memory is pinned.  On some systems, registered memory may not be
    actually be pinned but uses OS or hardware facilities to all GPU access to the host memory.
    Developers are strongly encouraged to register memory blocks which are aligned to the host
    cache-line size. (typically 64-bytes but can be obtains from the CPUID instruction).
    If registering non-aligned pointers, the application must take care when register pointers from
    the same cache line on different devices.  HIP's coarse-grained synchronization model does not
    guarantee correct results if different devices write to different parts of the same cache block -
    typically one of the writes will "win" and overwrite data from the other registered memory
    region.
    @return #hipSuccess, #hipErrorOutOfMemory
    @see hipHostUnregister, hipHostGetFlags, hipHostGetDevicePointer
    """
    pass

def hipHostUnregister(hostPtr) nogil:
    """@brief Un-register host pointer
    @param[in] hostPtr Host pointer previously registered with #hipHostRegister
    @return Error code
    @see hipHostRegister
    """
    pass

def hipMallocPitch(ptr,pitch,width,height) nogil:
    """Allocates at least width (in bytes) * height bytes of linear memory
    Padding may occur to ensure alighnment requirements are met for the given row
    The change in width size due to padding will be returned in *pitch.
    Currently the alignment is set to 128 bytes
    @param[out] ptr Pointer to the allocated device memory
    @param[out] pitch Pitch for allocation (in bytes)
    @param[in]  width Requested pitched allocation width (in bytes)
    @param[in]  height Requested pitched allocation height
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return Error code
    @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
    hipMalloc3DArray, hipHostMalloc
    """
    pass

def hipMemAllocPitch(dptr,pitch,widthInBytes,height,elementSizeBytes) nogil:
    """Allocates at least width (in bytes) * height bytes of linear memory
    Padding may occur to ensure alighnment requirements are met for the given row
    The change in width size due to padding will be returned in *pitch.
    Currently the alignment is set to 128 bytes
    @param[out] dptr Pointer to the allocated device memory
    @param[out] pitch Pitch for allocation (in bytes)
    @param[in]  width Requested pitched allocation width (in bytes)
    @param[in]  height Requested pitched allocation height
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array.
    Given the row and column of an array element of type T, the address is computed as:
    T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
    @return Error code
    @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
    hipMalloc3DArray, hipHostMalloc
    """
    pass

def hipFree(ptr) nogil:
    """@brief Free memory allocated by the hcc hip memory allocation API.
    This API performs an implicit hipDeviceSynchronize() call.
    If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
    @param[in] ptr Pointer to memory to be freed
    @return #hipSuccess
    @return #hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated
    with hipHostMalloc)
    @see hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
    hipMalloc3DArray, hipHostMalloc
    """
    pass

def hipFreeHost(ptr) nogil:
    """@brief Free memory allocated by the hcc hip host memory allocation API.  [Deprecated]
    @param[in] ptr Pointer to memory to be freed
    @return #hipSuccess,
    #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
     hipMalloc)
    @deprecated use hipHostFree() instead
    """
    pass

def hipHostFree(ptr) nogil:
    """@brief Free memory allocated by the hcc hip host memory allocation API
    This API performs an implicit hipDeviceSynchronize() call.
    If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
    @param[in] ptr Pointer to memory to be freed
    @return #hipSuccess,
    #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
    hipMalloc)
    @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D,
    hipMalloc3DArray, hipHostMalloc
    """
    pass

def hipMemcpy(dst,src,sizeBytes,kind) nogil:
    """@brief Copy data from src to dst.
    It supports memory from host to device,
    device to host, device to device and host to host
    The src and dst must not overlap.
    For hipMemcpy, the copy is always performed by the current device (set by hipSetDevice).
    For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the
    device where the src data is physically located. For optimal peer-to-peer copies, the copy device
    must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy
    agent as the current device and src/dest as the peerDevice argument.  if this is not done, the
    hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
    Calling hipMemcpy with dst and src pointers that do not match the hipMemcpyKind results in
    undefined behavior.
    @param[out]  dst Data being copy to
    @param[in]  src Data being copy from
    @param[in]  sizeBytes Data size in bytes
    @param[in]  copyType Memory copy type
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknowni
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyWithStream(dst,src,sizeBytes,kind,stream) nogil:
    """
    """
    pass

def hipMemcpyHtoD(dst,src,sizeBytes) nogil:
    """@brief Copy data from Host to Device
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyDtoH(dst,src,sizeBytes) nogil:
    """@brief Copy data from Device to Host
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyDtoD(dst,src,sizeBytes) nogil:
    """@brief Copy data from Device to Device
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyHtoDAsync(dst,src,sizeBytes,stream) nogil:
    """@brief Copy data from Host to Device asynchronously
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyDtoHAsync(dst,src,sizeBytes,stream) nogil:
    """@brief Copy data from Device to Host asynchronously
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyDtoDAsync(dst,src,sizeBytes,stream) nogil:
    """@brief Copy data from Device to Device asynchronously
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipModuleGetGlobal(dptr,bytes,hmod,name) nogil:
    """@brief Returns a global pointer from a module.
    Returns in *dptr and *bytes the pointer and size of the global of name name located in module hmod.
    If no variable of that name exists, it returns hipErrorNotFound. Both parameters dptr and bytes are optional.
    If one of them is NULL, it is ignored and hipSuccess is returned.
    @param[out]  dptr  Returns global device pointer
    @param[out]  bytes Returns global size in bytes
    @param[in]   hmod  Module to retrieve global from
    @param[in]   name  Name of global to retrieve
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotFound, #hipErrorInvalidContext
    """
    pass

def hipGetSymbolAddress(devPtr,symbol) nogil:
    """@brief Gets device pointer associated with symbol on the device.
    @param[out]  devPtr  pointer to the device associated the symbole
    @param[in]   symbol  pointer to the symbole of the device
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipGetSymbolSize(size,symbol) nogil:
    """@brief Gets the size of the given symbol on the device.
    @param[in]   symbol  pointer to the device symbole
    @param[out]  size  pointer to the size
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyToSymbol(symbol,src,sizeBytes,offset,kind) nogil:
    """@brief Copies data to the given symbol on the device.
    Symbol HIP APIs allow a kernel to define a device-side data symbol which can be accessed on
    the host side. The symbol can be in __constant or device space.
    Note that the symbol name needs to be encased in the HIP_SYMBOL macro.
    This also applies to hipMemcpyFromSymbol, hipGetSymbolAddress, and hipGetSymbolSize.
    For detail usage, see the example at
    https://github.com/ROCm-Developer-Tools/HIP/blob/rocm-5.0.x/docs/markdown/hip_porting_guide.md
    @param[out]  symbol  pointer to the device symbole
    @param[in]   src  pointer to the source address
    @param[in]   sizeBytes  size in bytes to copy
    @param[in]   offset  offset in bytes from start of symbole
    @param[in]   kind  type of memory transfer
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyToSymbolAsync(symbol,src,sizeBytes,offset,kind,stream) nogil:
    """@brief Copies data to the given symbol on the device asynchronously.
    @param[out]  symbol  pointer to the device symbole
    @param[in]   src  pointer to the source address
    @param[in]   sizeBytes  size in bytes to copy
    @param[in]   offset  offset in bytes from start of symbole
    @param[in]   kind  type of memory transfer
    @param[in]   stream  stream identifier
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyFromSymbol(dst,symbol,sizeBytes,offset,kind) nogil:
    """@brief Copies data from the given symbol on the device.
    @param[out]  dptr  Returns pointer to destinition memory address
    @param[in]   symbol  pointer to the symbole address on the device
    @param[in]   sizeBytes  size in bytes to copy
    @param[in]   offset  offset in bytes from the start of symbole
    @param[in]   kind  type of memory transfer
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyFromSymbolAsync(dst,symbol,sizeBytes,offset,kind,stream) nogil:
    """@brief Copies data from the given symbol on the device asynchronously.
    @param[out]  dptr  Returns pointer to destinition memory address
    @param[in]   symbol  pointer to the symbole address on the device
    @param[in]   sizeBytes  size in bytes to copy
    @param[in]   offset  offset in bytes from the start of symbole
    @param[in]   kind  type of memory transfer
    @param[in]   stream  stream identifier
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyAsync(dst,src,sizeBytes,kind,stream) nogil:
    """@brief Copy data from src to dst asynchronously.
    @warning If host or dest are not pinned, the memory copy will be performed synchronously.  For
    best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.
    @warning on HCC hipMemcpyAsync does not support overlapped H2D and D2H copies.
    For hipMemcpy, the copy is always performed by the device associated with the specified stream.
    For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is a
    attached to the device where the src data is physically located. For optimal peer-to-peer copies,
    the copy device must be able to access the src and dst pointers (by calling
    hipDeviceEnablePeerAccess with copy agent as the current device and src/dest as the peerDevice
    argument.  if this is not done, the hipMemcpy will still work, but will perform the copy using a
    staging buffer on the host.
    @param[out] dst Data being copy to
    @param[in]  src Data being copy from
    @param[in]  sizeBytes Data size in bytes
    @param[in]  accelerator_view Accelerator view which the copy is being enqueued
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
    @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyToSymbol,
    hipMemcpyFromSymbol, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync,
    hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync,
    hipMemcpyFromSymbolAsync
    """
    pass

def hipMemset(dst,value,sizeBytes) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    byte value value.
    @param[out] dst Data being filled
    @param[in]  constant value to be set
    @param[in]  sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD8(dest,value,count) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    byte value value.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD8Async(dest,value,count,stream) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    byte value value.
    hipMemsetD8Async() is asynchronous with respect to the host, so the call may return before the
    memset is complete. The operation can optionally be associated to a stream by passing a non-zero
    stream argument. If stream is non-zero, the operation may overlap with operations in other
    streams.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @param[in]  stream - Stream identifier
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD16(dest,value,count) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    short value value.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD16Async(dest,value,count,stream) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    short value value.
    hipMemsetD16Async() is asynchronous with respect to the host, so the call may return before the
    memset is complete. The operation can optionally be associated to a stream by passing a non-zero
    stream argument. If stream is non-zero, the operation may overlap with operations in other
    streams.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @param[in]  stream - Stream identifier
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD32(dest,value,count) nogil:
    """@brief Fills the memory area pointed to by dest with the constant integer
    value for specified number of times.
    @param[out] dst Data being filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetAsync(dst,value,sizeBytes,stream) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant
    byte value value.
    hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the
    memset is complete. The operation can optionally be associated to a stream by passing a non-zero
    stream argument. If stream is non-zero, the operation may overlap with operations in other
    streams.
    @param[out] dst Pointer to device memory
    @param[in]  value - Value to set for each byte of specified memory
    @param[in]  sizeBytes - Size in bytes to set
    @param[in]  stream - Stream identifier
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemsetD32Async(dst,value,count,stream) nogil:
    """@brief Fills the memory area pointed to by dev with the constant integer
    value for specified number of times.
    hipMemsetD32Async() is asynchronous with respect to the host, so the call may return before the
    memset is complete. The operation can optionally be associated to a stream by passing a non-zero
    stream argument. If stream is non-zero, the operation may overlap with operations in other
    streams.
    @param[out] dst Pointer to device memory
    @param[in]  value - Value to set for each byte of specified memory
    @param[in]  count - number of values to be set
    @param[in]  stream - Stream identifier
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset2D(dst,pitch,value,width,height) nogil:
    """@brief Fills the memory area pointed to by dst with the constant value.
    @param[out] dst Pointer to device memory
    @param[in]  pitch - data size in bytes
    @param[in]  value - constant value to be set
    @param[in]  width
    @param[in]  height
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset2DAsync(dst,pitch,value,width,height,stream) nogil:
    """@brief Fills asynchronously the memory area pointed to by dst with the constant value.
    @param[in]  dst Pointer to device memory
    @param[in]  pitch - data size in bytes
    @param[in]  value - constant value to be set
    @param[in]  width
    @param[in]  height
    @param[in]  stream
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset3D(pitchedDevPtr,value,extent) nogil:
    """@brief Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
    @param[in] pitchedDevPtr
    @param[in]  value - constant value to be set
    @param[in]  extent
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset3DAsync(pitchedDevPtr,value,extent,stream) nogil:
    """@brief Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
    @param[in] pitchedDevPtr
    @param[in]  value - constant value to be set
    @param[in]  extent
    @param[in]  stream
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemGetInfo(free,total) nogil:
    """@brief Query memory info.
    Return snapshot of free memory, and total allocatable memory on the device.
    Returns in *free a snapshot of the current free memory.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @warning On HCC, the free memory only accounts for memory allocated by this process and may be
    optimistic.
    """
    pass

def hipMemPtrGetInfo(ptr,size) nogil:
    """
    """
    pass

def hipMallocArray(array,desc,width,height,flags) nogil:
    """@brief Allocate an array on the device.
    @param[out]  array  Pointer to allocated array in device memory
    @param[in]   desc   Requested channel format
    @param[in]   width  Requested array allocation width
    @param[in]   height Requested array allocation height
    @param[in]   flags  Requested properties of allocated array
    @return      #hipSuccess, #hipErrorOutOfMemory
    @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
    """
    pass

def hipArrayCreate(pHandle,pAllocateArray) nogil:
    """
    """
    pass

def hipArrayDestroy(array) nogil:
    """
    """
    pass

def hipArray3DCreate(array,pAllocateArray) nogil:
    """
    """
    pass

def hipMalloc3D(pitchedDevPtr,extent) nogil:
    """
    """
    pass

def hipFreeArray(array) nogil:
    """@brief Frees an array on the device.
    @param[in]  array  Pointer to array to free
    @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
    """
    pass

def hipFreeMipmappedArray(mipmappedArray) nogil:
    """@brief Frees a mipmapped array on the device
    @param[in] mipmappedArray - Pointer to mipmapped array to free
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMalloc3DArray(array,desc,extent,flags) nogil:
    """@brief Allocate an array on the device.
    @param[out]  array  Pointer to allocated array in device memory
    @param[in]   desc   Requested channel format
    @param[in]   extent Requested array allocation width, height and depth
    @param[in]   flags  Requested properties of allocated array
    @return      #hipSuccess, #hipErrorOutOfMemory
    @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
    """
    pass

def hipMallocMipmappedArray(mipmappedArray,desc,extent,numLevels,flags) nogil:
    """@brief Allocate a mipmapped array on the device
    @param[out] mipmappedArray  - Pointer to allocated mipmapped array in device memory
    @param[in]  desc            - Requested channel format
    @param[in]  extent          - Requested allocation size (width field in elements)
    @param[in]  numLevels       - Number of mipmap levels to allocate
    @param[in]  flags           - Flags for extensions
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    """
    pass

def hipGetMipmappedArrayLevel(levelArray,mipmappedArray,level) nogil:
    """@brief Gets a mipmap level of a HIP mipmapped array
    @param[out] levelArray     - Returned mipmap level HIP array
    @param[in]  mipmappedArray - HIP mipmapped array
    @param[in]  level          - Mipmap level
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpy2D(dst,dpitch,src,spitch,width,height,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst    Destination memory address
    @param[in]   dpitch Pitch of destination memory
    @param[in]   src    Source memory address
    @param[in]   spitch Pitch of source memory
    @param[in]   width  Width of matrix transfer (columns in bytes)
    @param[in]   height Height of matrix transfer (rows)
    @param[in]   kind   Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyParam2D(pCopy) nogil:
    """@brief Copies memory for 2D arrays.
    @param[in]   pCopy Parameters for the memory copy
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    hipMemcpyToSymbol, hipMemcpyAsync
    """
    pass

def hipMemcpyParam2DAsync(pCopy,stream) nogil:
    """@brief Copies memory for 2D arrays.
    @param[in]   pCopy Parameters for the memory copy
    @param[in]   stream Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    hipMemcpyToSymbol, hipMemcpyAsync
    """
    pass

def hipMemcpy2DAsync(dst,dpitch,src,spitch,width,height,kind,stream) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst    Destination memory address
    @param[in]   dpitch Pitch of destination memory
    @param[in]   src    Source memory address
    @param[in]   spitch Pitch of source memory
    @param[in]   width  Width of matrix transfer (columns in bytes)
    @param[in]   height Height of matrix transfer (rows)
    @param[in]   kind   Type of transfer
    @param[in]   stream Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy2DToArray(dst,wOffset,hOffset,src,spitch,width,height,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst     Destination memory address
    @param[in]   wOffset Destination starting X offset
    @param[in]   hOffset Destination starting Y offset
    @param[in]   src     Source memory address
    @param[in]   spitch  Pitch of source memory
    @param[in]   width   Width of matrix transfer (columns in bytes)
    @param[in]   height  Height of matrix transfer (rows)
    @param[in]   kind    Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy2DToArrayAsync(dst,wOffset,hOffset,src,spitch,width,height,kind,stream) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst     Destination memory address
    @param[in]   wOffset Destination starting X offset
    @param[in]   hOffset Destination starting Y offset
    @param[in]   src     Source memory address
    @param[in]   spitch  Pitch of source memory
    @param[in]   width   Width of matrix transfer (columns in bytes)
    @param[in]   height  Height of matrix transfer (rows)
    @param[in]   kind    Type of transfer
    @param[in]   stream    Accelerator view which the copy is being enqueued
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyToArray(dst,wOffset,hOffset,src,count,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst     Destination memory address
    @param[in]   wOffset Destination starting X offset
    @param[in]   hOffset Destination starting Y offset
    @param[in]   src     Source memory address
    @param[in]   count   size in bytes to copy
    @param[in]   kind    Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyFromArray(dst,srcArray,wOffset,hOffset,count,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst       Destination memory address
    @param[in]   srcArray  Source memory address
    @param[in]   woffset   Source starting X offset
    @param[in]   hOffset   Source starting Y offset
    @param[in]   count     Size in bytes to copy
    @param[in]   kind      Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy2DFromArray(dst,dpitch,src,wOffset,hOffset,width,height,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst       Destination memory address
    @param[in]   dpitch    Pitch of destination memory
    @param[in]   src       Source memory address
    @param[in]   wOffset   Source starting X offset
    @param[in]   hOffset   Source starting Y offset
    @param[in]   width     Width of matrix transfer (columns in bytes)
    @param[in]   height    Height of matrix transfer (rows)
    @param[in]   kind      Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy2DFromArrayAsync(dst,dpitch,src,wOffset,hOffset,width,height,kind,stream) nogil:
    """@brief Copies data between host and device asynchronously.
    @param[in]   dst       Destination memory address
    @param[in]   dpitch    Pitch of destination memory
    @param[in]   src       Source memory address
    @param[in]   wOffset   Source starting X offset
    @param[in]   hOffset   Source starting Y offset
    @param[in]   width     Width of matrix transfer (columns in bytes)
    @param[in]   height    Height of matrix transfer (rows)
    @param[in]   kind      Type of transfer
    @param[in]   stream    Accelerator view which the copy is being enqueued
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyAtoH(dst,srcArray,srcOffset,count) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst       Destination memory address
    @param[in]   srcArray  Source array
    @param[in]   srcoffset Offset in bytes of source array
    @param[in]   count     Size of memory copy in bytes
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyHtoA(dstArray,dstOffset,srcHost,count) nogil:
    """@brief Copies data between host and device.
    @param[in]   dstArray   Destination memory address
    @param[in]   dstOffset  Offset in bytes of destination array
    @param[in]   srcHost    Source host pointer
    @param[in]   count      Size of memory copy in bytes
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy3D(p) nogil:
    """@brief Copies data between host and device.
    @param[in]   p   3D memory copy parameters
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy3DAsync(p,stream) nogil:
    """@brief Copies data between host and device asynchronously.
    @param[in]   p        3D memory copy parameters
    @param[in]   stream   Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipDrvMemcpy3D(pCopy) nogil:
    """@brief Copies data between host and device.
    @param[in]   pCopy   3D memory copy parameters
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipDrvMemcpy3DAsync(pCopy,stream) nogil:
    """@brief Copies data between host and device asynchronously.
    @param[in]   pCopy    3D memory copy parameters
    @param[in]   stream   Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipDeviceCanAccessPeer(canAccessPeer,deviceId,peerDeviceId) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup PeerToPeer PeerToPeer Device Memory Access
    @{
    @warning PeerToPeer support is experimental.
    This section describes the PeerToPeer device memory access functions of HIP runtime API.
    @brief Determine if a device can access a peer's memory.
    @param [out] canAccessPeer Returns the peer access capability (0 or 1)
    @param [in] device - device from where memory may be accessed.
    @param [in] peerDevice - device where memory is physically located
    Returns "1" in @p canAccessPeer if the specified @p device is capable
    of directly accessing memory physically located on peerDevice , or "0" if not.
    Returns "0" in @p canAccessPeer if deviceId == peerDeviceId, and both are valid devices : a
    device is not a peer of itself.
    @returns #hipSuccess,
    @returns #hipErrorInvalidDevice if deviceId or peerDeviceId are not valid devices
    """
    pass

def hipDeviceEnablePeerAccess(peerDeviceId,flags) nogil:
    """@brief Enable direct access from current device's virtual address space to memory allocations
    physically located on a peer device.
    Memory which already allocated on peer device will be mapped into the address space of the
    current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
    the address space of the current device when the memory is allocated. The peer memory remains
    accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
    @param [in] peerDeviceId
    @param [in] flags
    Returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
    @returns #hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device.
    """
    pass

def hipDeviceDisablePeerAccess(peerDeviceId) nogil:
    """@brief Disable direct access from current device's virtual address space to memory allocations
    physically located on a peer device.
    Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
    enabled from the current device.
    @param [in] peerDeviceId
    @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
    """
    pass

def hipMemGetAddressRange(pbase,psize,dptr) nogil:
    """@brief Get information on memory allocations.
    @param [out] pbase - BAse pointer address
    @param [out] psize - Size of allocation
    @param [in]  dptr- Device Pointer
    @returns #hipSuccess, #hipErrorInvalidDevicePointer
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipMemcpyPeer(dst,dstDeviceId,src,srcDeviceId,sizeBytes) nogil:
    """@brief Copies memory from one device to memory on another device.
    @param [out] dst - Destination device pointer.
    @param [in] dstDeviceId - Destination device
    @param [in] src - Source device pointer
    @param [in] srcDeviceId - Source device
    @param [in] sizeBytes - Size of memory copy in bytes
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
    """
    pass

def hipMemcpyPeerAsync(dst,dstDeviceId,src,srcDevice,sizeBytes,stream) nogil:
    """@brief Copies memory from one device to memory on another device.
    @param [out] dst - Destination device pointer.
    @param [in] dstDevice - Destination device
    @param [in] src - Source device pointer
    @param [in] srcDevice - Source device
    @param [in] sizeBytes - Size of memory copy in bytes
    @param [in] stream - Stream identifier
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
    """
    pass

def hipCtxCreate(ctx,flags,device) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Context Context Management
    @{
    This section describes the context management functions of HIP runtime API.
    @addtogroup ContextD Context Management [Deprecated]
    @{
    @ingroup Context
    This section describes the deprecated context management functions of HIP runtime API.
    @brief Create a context and set it as current/ default context
    @param [out] ctx
    @param [in] flags
    @param [in] associated device handle
    @return #hipSuccess
    @see hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent,
    hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxDestroy(ctx) nogil:
    """@brief Destroy a HIP context.
    @param [in] ctx Context to destroy
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipCtxCreate, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,hipCtxSetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    """
    pass

def hipCtxPopCurrent(ctx) nogil:
    """@brief Pop the current/default context and return the popped context.
    @param [out] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxSetCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxPushCurrent(ctx) nogil:
    """@brief Push the context to be set as current/ default context
    @param [in] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    """
    pass

def hipCtxSetCurrent(ctx) nogil:
    """@brief Set the passed context as current/default
    @param [in] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    """
    pass

def hipCtxGetCurrent(ctx) nogil:
    """@brief Get the handle of the current/ default context
    @param [out] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxGetDevice(device) nogil:
    """@brief Get the handle of the device associated with current/default context
    @param [out] device
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize
    """
    pass

def hipCtxGetApiVersion(ctx,apiVersion) nogil:
    """@brief Returns the approximate HIP api version.
    @param [in]  ctx Context to check
    @param [out] apiVersion
    @return #hipSuccess
    @warning The HIP feature set does not correspond to an exact CUDA SDK api revision.
    This function always set *apiVersion to 4 as an approximation though HIP supports
    some features which were introduced in later CUDA SDK revisions.
    HIP apps code should not rely on the api revision number here and should
    use arch feature flags to test device capabilities or conditional compilation.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxGetCacheConfig(cacheConfig) nogil:
    """@brief Set Cache configuration for a specific function
    @param [out] cacheConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxSetCacheConfig(cacheConfig) nogil:
    """@brief Set L1/Shared cache partition.
    @param [in] cacheConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxSetSharedMemConfig(config) nogil:
    """@brief Set Shared memory bank configuration.
    @param [in] sharedMemoryConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxGetSharedMemConfig(pConfig) nogil:
    """@brief Get Shared memory bank configuration.
    @param [out] sharedMemoryConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxSynchronize() nogil:
    """@brief Blocks until the default context has completed all preceding requested tasks.
    @return #hipSuccess
    @warning This function waits for all streams on the default context to complete execution, and
    then returns.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxGetDevice
    """
    pass

def hipCtxGetFlags(flags) nogil:
    """@brief Return flags used for creating default context.
    @param [out] flags
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxEnablePeerAccess(peerCtx,flags) nogil:
    """@brief Enables direct access to memory allocations in a peer context.
    Memory which already allocated on peer device will be mapped into the address space of the
    current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
    the address space of the current device when the memory is allocated. The peer memory remains
    accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
    @param [in] peerCtx
    @param [in] flags
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
    #hipErrorPeerAccessAlreadyEnabled
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    @warning PeerToPeer support is experimental.
    """
    pass

def hipCtxDisablePeerAccess(peerCtx) nogil:
    """@brief Disable direct access from current context's virtual address space to memory allocations
    physically located on a peer context.Disables direct access to memory allocations in a peer
    context and unregisters any registered allocations.
    Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
    enabled from the current device.
    @param [in] peerCtx
    @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    @warning PeerToPeer support is experimental.
    """
    pass

def hipDevicePrimaryCtxGetState(dev,flags,active) nogil:
    """@}
    @brief Get the state of the primary context.
    @param [in] Device to get primary context flags for
    @param [out] Pointer to store flags
    @param [out] Pointer to store context state; 0 = inactive, 1 = active
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipDevicePrimaryCtxRelease(dev) nogil:
    """@brief Release the primary context on the GPU.
    @param [in] Device which primary context is released
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    @warning This function return #hipSuccess though doesn't release the primaryCtx by design on
    HIP/HCC path.
    """
    pass

def hipDevicePrimaryCtxRetain(pctx,dev) nogil:
    """@brief Retain the primary context on the GPU.
    @param [out] Returned context handle of the new context
    @param [in] Device which primary context is released
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipDevicePrimaryCtxReset(dev) nogil:
    """@brief Resets the primary context on the GPU.
    @param [in] Device which primary context is reset
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipDevicePrimaryCtxSetFlags(dev,flags) nogil:
    """@brief Set flags for the primary context.
    @param [in] Device for which the primary context flags are set
    @param [in] New flags for the device
    @returns #hipSuccess, #hipErrorContextAlreadyInUse
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipModuleLoad(module,fname) nogil:
    """@}
    @defgroup Module Module Management
    @{
    This section describes the module management functions of HIP runtime API.
    @brief Loads code object from file into a hipModule_t
    @param [in] fname
    @param [out] module
    @warning File/memory resources allocated in this function are released only in hipModuleUnload.
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorFileNotFound,
    hipErrorOutOfMemory, hipErrorSharedObjectInitFailed, hipErrorNotInitialized
    """
    pass

def hipModuleUnload(module) nogil:
    """@brief Frees the module
    @param [in] module
    @returns hipSuccess, hipInvalidValue
    module is freed and the code objects associated with it are destroyed
    """
    pass

def hipModuleGetFunction(function,module,kname) nogil:
    """@brief Function with kname will be extracted if present in module
    @param [in] module
    @param [in] kname
    @param [out] function
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorNotInitialized,
    hipErrorNotFound,
    """
    pass

def hipFuncGetAttributes(attr,func) nogil:
    """@brief Find out attributes for a given function.
    @param [out] attr
    @param [in] func
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
    """
    pass

def hipFuncGetAttribute(value,attrib,hfunc) nogil:
    """@brief Find out a specific attribute for a given function.
    @param [out] value
    @param [in]  attrib
    @param [in]  hfunc
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
    """
    pass

def hipModuleGetTexRef(texRef,hmod,name) nogil:
    """@brief returns the handle of the texture reference with the name from the module.
    @param [in] hmod
    @param [in] name
    @param [out] texRef
    @returns hipSuccess, hipErrorNotInitialized, hipErrorNotFound, hipErrorInvalidValue
    """
    pass

def hipModuleLoadData(module,image) nogil:
    """@brief builds module from code object which resides in host memory. Image is pointer to that
    location.
    @param [in] image
    @param [out] module
    @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
    """
    pass

def hipModuleLoadDataEx(module,image,numOptions,options,optionValues) nogil:
    """@brief builds module from code object which resides in host memory. Image is pointer to that
    location. Options are not used. hipModuleLoadData is called.
    @param [in] image
    @param [out] module
    @param [in] number of options
    @param [in] options for JIT
    @param [in] option values for JIT
    @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
    """
    pass

def hipModuleLaunchKernel(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,stream,kernelParams,extra) nogil:
    """@brief launches kernel f with launch parameters and shared memory on stream with arguments passed
    to kernelparams or extra
    @param [in] f         Kernel to launch.
    @param [in] gridDimX  X grid dimension specified as multiple of blockDimX.
    @param [in] gridDimY  Y grid dimension specified as multiple of blockDimY.
    @param [in] gridDimZ  Z grid dimension specified as multiple of blockDimZ.
    @param [in] blockDimX X block dimensions specified in work-items
    @param [in] blockDimY Y grid dimension specified in work-items
    @param [in] blockDimZ Z grid dimension specified in work-items
    @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
    default stream is used with associated synchronization rules.
    @param [in] kernelParams
    @param [in] extra     Pointer to kernel arguments.   These are passed directly to the kernel and
    must be in the memory layout and alignment expected by the kernel.
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32. So gridDim.x * blockDim.x, gridDim.y * blockDim.y
    and gridDim.z * blockDim.z are always less than 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    @warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please
    refer to hip_porting_driver_api.md for sample usage.
    """
    pass

def hipLaunchCooperativeKernel(f,gridDim,blockDimX,kernelParams,sharedMemBytes,stream) nogil:
    """@brief launches kernel f with launch parameters and shared memory on stream with arguments passed
    to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute
    @param [in] f         Kernel to launch.
    @param [in] gridDim   Grid dimensions specified as multiple of blockDim.
    @param [in] blockDim  Block dimensions specified in work-items
    @param [in] kernelParams A list of kernel arguments
    @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
    default stream is used with associated synchronization rules.
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
    """
    pass

def hipLaunchCooperativeKernelMultiDevice(launchParamsList,numDevices,flags) nogil:
    """@brief Launches kernels on multiple devices where thread blocks can cooperate and
    synchronize as they execute.
    @param [in] launchParamsList         List of launch parameters, one per device.
    @param [in] numDevices               Size of the launchParamsList array.
    @param [in] flags                    Flags to control launch behavior.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
    """
    pass

def hipExtLaunchMultiKernelMultiDevice(launchParamsList,numDevices,flags) nogil:
    """@brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
    on respective streams before enqueuing any other work on the specified streams from any other threads
    @param [in] hipLaunchParams          List of launch parameters, one per device.
    @param [in] numDevices               Size of the launchParamsList array.
    @param [in] flags                    Flags to control launch behavior.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    """
    pass

def hipModuleOccupancyMaxPotentialBlockSize(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit) nogil:
    """@}
    @defgroup Occupancy Occupancy
    @{
    This section describes the occupancy functions of HIP runtime API.
    @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
    @param [out] gridSize           minimum grid size for maximum potential occupancy
    @param [out] blockSize          block size for maximum potential occupancy
    @param [in]  f                  kernel function for which occupancy is calulated
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
    """
    pass

def hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit,flags) nogil:
    """@brief determine the grid and block sizes to achieves maximum occupancy for a kernel
    @param [out] gridSize           minimum grid size for maximum potential occupancy
    @param [out] blockSize          block size for maximum potential occupancy
    @param [in]  f                  kernel function for which occupancy is calulated
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
    @param [in]  flags            Extra flags for occupancy calculation (only default supported)
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
    """
    pass

def hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks,f,blockSize,dynSharedMemPerBlk) nogil:
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  func             Kernel function (hipFunction) for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    """
    pass

def hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks,f,blockSize,dynSharedMemPerBlk,flags) nogil:
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  f                Kernel function(hipFunction_t) for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  flags            Extra flags for occupancy calculation (only default supported)
    """
    pass

def hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks,f,blockSize,dynSharedMemPerBlk) nogil:
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  func             Kernel function for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    """
    pass

def hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks,f,blockSize,dynSharedMemPerBlk,flags) nogil:
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  f                Kernel function for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  flags            Extra flags for occupancy calculation (currently ignored)
    """
    pass

def hipOccupancyMaxPotentialBlockSize(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit) nogil:
    """@brief determine the grid and block sizes to achieves maximum occupancy for a kernel
    @param [out] gridSize           minimum grid size for maximum potential occupancy
    @param [out] blockSize          block size for maximum potential occupancy
    @param [in]  f                  kernel function for which occupancy is calulated
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
    """
    pass

def hipProfilerStart() nogil:
    """@brief Start recording of profiling information
    When using this API, start the profiler with profiling disabled.  (--startdisabled)
    @warning : hipProfilerStart API is under development.
    """
    pass

def hipProfilerStop() nogil:
    """@brief Stop recording of profiling information.
    When using this API, start the profiler with profiling disabled.  (--startdisabled)
    @warning : hipProfilerStop API is under development.
    """
    pass

def hipConfigureCall(gridDim,blockDim,sharedMem,stream) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Clang Launch API to support the triple-chevron syntax
    @{
    This section describes the API to support the triple-chevron syntax.
    @brief Configure a kernel launch.
    @param [in] gridDim   grid dimension specified as multiple of blockDim.
    @param [in] blockDim  block dimensions specified in work-items
    @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
    default stream is used with associated synchronization rules.
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    """
    pass

def hipSetupArgument(arg,size,offset) nogil:
    """@brief Set a kernel argument.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    @param [in] arg    Pointer the argument in host memory.
    @param [in] size   Size of the argument.
    @param [in] offset Offset of the argument on the argument stack.
    """
    pass

def hipLaunchByPtr(func) nogil:
    """@brief Launch a kernel.
    @param [in] func Kernel to launch.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    """
    pass

def hipLaunchKernel(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream) nogil:
    """@brief C compliant kernel launch API
    @param [in] function_address - kernel stub function pointer.
    @param [in] numBlocks - number of blocks
    @param [in] dimBlocks - dimension of a block
    @param [in] args - kernel arguments
    @param [in] sharedMemBytes - Amount of dynamic shared memory to allocate for this kernel. The
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream - Stream where the kernel should be dispatched.  May be 0, in which case th
    default stream is used with associated synchronization rules.
    @returns #hipSuccess, #hipErrorInvalidValue, hipInvalidDevice
    """
    pass

def hipLaunchHostFunc(stream,fn,userData) nogil:
    """@brief Enqueues a host function call in a stream.
    @param [in] stream - stream to enqueue work to.
    @param [in] fn - function to call once operations enqueued preceeding are complete.
    @param [in] userData - User-specified data to be passed to the function.
    @returns #hipSuccess, #hipErrorInvalidResourceHandle, #hipErrorInvalidValue,
    #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDrvMemcpy2DUnaligned(pCopy) nogil:
    """Copies memory for 2D arrays.
    @param pCopy           - Parameters for the memory copy
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipExtLaunchKernel(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream,startEvent,stopEvent,flags) nogil:
    """@brief Launches kernel from the pointer address, with arguments and shared memory on stream.
    @param [in] function_address pointer to the Kernel to launch.
    @param [in] numBlocks number of blocks.
    @param [in] dimBlocks dimension of a block.
    @param [in] args pointer to kernel arguments.
    @param [in] sharedMemBytes  Amount of dynamic shared memory to allocate for this kernel.
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream  Stream where the kernel should be dispatched.
    @param [in] startEvent  If non-null, specified event will be updated to track the start time of
    the kernel launch. The event must be created before calling this API.
    @param [in] stopEvent  If non-null, specified event will be updated to track the stop time of
    the kernel launch. The event must be created before calling this API.
    May be 0, in which case the default stream is used with associated synchronization rules.
    @param [in] flags. The value of hipExtAnyOrderLaunch, signifies if kernel can be
    launched in any order.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue.
    """
    pass

def hipBindTextureToMipmappedArray(tex,mipmappedArray,desc) nogil:
    """@brief  Binds a mipmapped array to a texture.
    @param [in] tex  pointer to the texture reference to bind
    @param [in] mipmappedArray  memory mipmapped array on the device
    @param [in] desc  opointer to the channel format
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipCreateTextureObject(pTexObject,pResDesc,pTexDesc,pResViewDesc) nogil:
    """@brief Creates a texture object.
    @param [out] pTexObject  pointer to the texture object to create
    @param [in] pResDesc  pointer to resource descriptor
    @param [in] pTexDesc  pointer to texture descriptor
    @param [in] pResViewDesc  pointer to resource view descriptor
    @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported, hipErrorOutOfMemory
    @note 3D liner filter isn't supported on GFX90A boards, on which the API @p hipCreateTextureObject will
    return hipErrorNotSupported.
    """
    pass

def hipDestroyTextureObject(textureObject) nogil:
    """@brief Destroys a texture object.
    @param [in] textureObject  texture object to destroy
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetChannelDesc(desc,array) nogil:
    """@brief Gets the channel descriptor in an array.
    @param [in] desc  pointer to channel format descriptor
    @param [out] array  memory array on the device
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetTextureObjectResourceDesc(pResDesc,textureObject) nogil:
    """@brief Gets resource descriptor for the texture object.
    @param [out] pResDesc  pointer to resource descriptor
    @param [in] textureObject  texture object
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetTextureObjectResourceViewDesc(pResViewDesc,textureObject) nogil:
    """@brief Gets resource view descriptor for the texture object.
    @param [out] pResViewDesc  pointer to resource view descriptor
    @param [in] textureObject  texture object
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetTextureObjectTextureDesc(pTexDesc,textureObject) nogil:
    """@brief Gets texture descriptor for the texture object.
    @param [out] pTexDesc  pointer to texture descriptor
    @param [in] textureObject  texture object
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexObjectCreate(pTexObject,pResDesc,pTexDesc,pResViewDesc) nogil:
    """@brief Creates a texture object.
    @param [out] pTexObject  pointer to texture object to create
    @param [in] pResDesc  pointer to resource descriptor
    @param [in] pTexDesc  pointer to texture descriptor
    @param [in] pResViewDesc  pointer to resource view descriptor
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexObjectDestroy(texObject) nogil:
    """@brief Destroys a texture object.
    @param [in] texObject  texture object to destroy
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexObjectGetResourceDesc(pResDesc,texObject) nogil:
    """@brief Gets resource descriptor of a texture object.
    @param [out] pResDesc  pointer to resource descriptor
    @param [in] texObject  texture object
    @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    """
    pass

def hipTexObjectGetResourceViewDesc(pResViewDesc,texObject) nogil:
    """@brief Gets resource view descriptor of a texture object.
    @param [out] pResViewDesc  pointer to resource view descriptor
    @param [in] texObject  texture object
    @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    """
    pass

def hipTexObjectGetTextureDesc(pTexDesc,texObject) nogil:
    """@brief Gets texture descriptor of a texture object.
    @param [out] pTexDesc  pointer to texture descriptor
    @param [in] texObject  texture object
    @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    """
    pass

def hipGetTextureReference(texref,symbol) nogil:
    """@addtogroup TextureD Texture Management [Deprecated]
    @{
    @ingroup Texture
    This section describes the deprecated texture management functions of HIP runtime API.
    @brief Gets the texture reference related with the symbol.
    @param [out] texref  texture reference
    @param [in] symbol  pointer to the symbol related with the texture for the reference
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexRefSetAddressMode(texRef,dim,am) nogil:
    """
    """
    pass

def hipTexRefSetArray(tex,array,flags) nogil:
    """
    """
    pass

def hipTexRefSetFilterMode(texRef,fm) nogil:
    """
    """
    pass

def hipTexRefSetFlags(texRef,Flags) nogil:
    """
    """
    pass

def hipTexRefSetFormat(texRef,fmt,NumPackedComponents) nogil:
    """
    """
    pass

def hipBindTexture(offset,tex,devPtr,desc,size) nogil:
    """
    """
    pass

def hipBindTexture2D(offset,tex,devPtr,desc,width,height,pitch) nogil:
    """
    """
    pass

def hipBindTextureToArray(tex,array,desc) nogil:
    """
    """
    pass

def hipGetTextureAlignmentOffset(offset,texref) nogil:
    """
    """
    pass

def hipUnbindTexture(tex) nogil:
    """
    """
    pass

def hipTexRefGetAddress(dev_ptr,texRef) nogil:
    """
    """
    pass

def hipTexRefGetAddressMode(pam,texRef,dim) nogil:
    """
    """
    pass

def hipTexRefGetFilterMode(pfm,texRef) nogil:
    """
    """
    pass

def hipTexRefGetFlags(pFlags,texRef) nogil:
    """
    """
    pass

def hipTexRefGetFormat(pFormat,pNumChannels,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMaxAnisotropy(pmaxAnsio,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMipmapFilterMode(pfm,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMipmapLevelBias(pbias,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp,pmaxMipmapLevelClamp,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMipMappedArray(pArray,texRef) nogil:
    """
    """
    pass

def hipTexRefSetAddress(ByteOffset,texRef,dptr,bytes) nogil:
    """
    """
    pass

def hipTexRefSetAddress2D(texRef,desc,dptr,Pitch) nogil:
    """
    """
    pass

def hipTexRefSetMaxAnisotropy(texRef,maxAniso) nogil:
    """
    """
    pass

def hipTexRefSetBorderColor(texRef,pBorderColor) nogil:
    """
    """
    pass

def hipTexRefSetMipmapFilterMode(texRef,fm) nogil:
    """
    """
    pass

def hipTexRefSetMipmapLevelBias(texRef,bias) nogil:
    """
    """
    pass

def hipTexRefSetMipmapLevelClamp(texRef,minMipMapLevelClamp,maxMipMapLevelClamp) nogil:
    """
    """
    pass

def hipTexRefSetMipmappedArray(texRef,mipmappedArray,Flags) nogil:
    """
    """
    pass

def hipMipmappedArrayCreate(pHandle,pMipmappedArrayDesc,numMipmapLevels) nogil:
    """@addtogroup TextureU Texture Management [Not supported]
    @{
    @ingroup Texture
    This section describes the texture management functions currently unsupported in HIP runtime.
    """
    pass

def hipMipmappedArrayDestroy(hMipmappedArray) nogil:
    """
    """
    pass

def hipMipmappedArrayGetLevel(pLevelArray,hMipMappedArray,level) nogil:
    """
    """
    pass

def hipApiName(id) nogil:
    """@defgroup Callback Callback Activity APIs
    @{
    This section describes the callback/Activity of HIP runtime API.
    """
    pass

def hipKernelNameRef(f) nogil:
    """
    """
    pass

def hipKernelNameRefByPtr(hostFunction,stream) nogil:
    """
    """
    pass

def hipGetStreamDeviceId(stream) nogil:
    """
    """
    pass

def hipStreamBeginCapture(stream,mode) nogil:
    """@brief Begins graph capture on a stream.
    @param [in] stream - Stream to initiate capture.
    @param [in] mode - Controls the interaction of this capture sequence with other API calls that
    are not safe.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamEndCapture(stream,pGraph) nogil:
    """@brief Ends capture on a stream, returning the captured graph.
    @param [in] stream - Stream to end capture.
    @param [out] pGraph - returns the graph captured.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamGetCaptureInfo(stream,pCaptureStatus,pId) nogil:
    """@brief Get capture status of a stream.
    @param [in] stream - Stream under capture.
    @param [out] pCaptureStatus - returns current status of the capture.
    @param [out] pId - unique ID of the capture.
    @returns #hipSuccess, #hipErrorStreamCaptureImplicit
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamGetCaptureInfo_v2(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out) nogil:
    """@brief Get stream's capture state
    @param [in] stream - Stream under capture.
    @param [out] captureStatus_out - returns current status of the capture.
    @param [out] id_out - unique ID of the capture.
    @param [in] graph_out - returns the graph being captured into.
    @param [out] dependencies_out - returns pointer to an array of nodes.
    @param [out] numDependencies_out - returns size of the array returned in dependencies_out.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamIsCapturing(stream,pCaptureStatus) nogil:
    """@brief Get stream's capture state
    @param [in] stream - Stream under capture.
    @param [out] pCaptureStatus - returns current status of the capture.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamUpdateCaptureDependencies(stream,dependencies,numDependencies,flags) nogil:
    """@brief Update the set of dependencies in a capturing stream
    @param [in] stream - Stream under capture.
    @param [in] dependencies - pointer to an array of nodes to Add/Replace.
    @param [in] numDependencies - size of the array in dependencies.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorIllegalState
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipThreadExchangeStreamCaptureMode(mode) nogil:
    """@brief Swaps the stream capture mode of a thread.
    @param [in] mode - Pointer to mode value to swap with the current mode
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphCreate(pGraph,flags) nogil:
    """@brief Creates a graph
    @param [out] pGraph - pointer to graph to create.
    @param [in] flags - flags for graph creation, must be 0.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphDestroy(graph) nogil:
    """@brief Destroys a graph
    @param [in] graph - instance of graph to destroy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddDependencies(graph,from,to,numDependencies) nogil:
    """@brief Adds dependency edges to a graph.
    @param [in] graph - instance of the graph to add dependencies.
    @param [in] from - pointer to the graph nodes with dependenties to add from.
    @param [in] to - pointer to the graph nodes to add dependenties to.
    @param [in] numDependencies - the number of dependencies to add.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphRemoveDependencies(graph,from,to,numDependencies) nogil:
    """@brief Removes dependency edges from a graph.
    @param [in] graph - instance of the graph to remove dependencies.
    @param [in] from - Array of nodes that provide the dependencies.
    @param [in] to - Array of dependent nodes.
    @param [in] numDependencies - the number of dependencies to remove.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphGetEdges(graph,from,to,numEdges) nogil:
    """@brief Returns a graph's dependency edges.
    @param [in] graph - instance of the graph to get the edges from.
    @param [out] from - pointer to the graph nodes to return edge endpoints.
    @param [out] to - pointer to the graph nodes to return edge endpoints.
    @param [out] numEdges - returns number of edges.
    @returns #hipSuccess, #hipErrorInvalidValue
    from and to may both be NULL, in which case this function only returns the number of edges in
    numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher than the actual
    number of edges, the remaining entries in from and to will be set to NULL, and the number of
    edges actually returned will be written to numEdges
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphGetNodes(graph,nodes,numNodes) nogil:
    """@brief Returns graph nodes.
    @param [in] graph - instance of graph to get the nodes.
    @param [out] nodes - pointer to return the  graph nodes.
    @param [out] numNodes - returns number of graph nodes.
    @returns #hipSuccess, #hipErrorInvalidValue
    nodes may be NULL, in which case this function will return the number of nodes in numNodes.
    Otherwise, numNodes entries will be filled in. If numNodes is higher than the actual number of
    nodes, the remaining entries in nodes will be set to NULL, and the number of nodes actually
    obtained will be returned in numNodes.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphGetRootNodes(graph,pRootNodes,pNumRootNodes) nogil:
    """@brief Returns graph's root nodes.
    @param [in] graph - instance of the graph to get the nodes.
    @param [out] pRootNodes - pointer to return the graph's root nodes.
    @param [out] pNumRootNodes - returns the number of graph's root nodes.
    @returns #hipSuccess, #hipErrorInvalidValue
    pRootNodes may be NULL, in which case this function will return the number of root nodes in
    pNumRootNodes. Otherwise, pNumRootNodes entries will be filled in. If pNumRootNodes is higher
    than the actual number of root nodes, the remaining entries in pRootNodes will be set to NULL,
    and the number of nodes actually obtained will be returned in pNumRootNodes.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphNodeGetDependencies(node,pDependencies,pNumDependencies) nogil:
    """@brief Returns a node's dependencies.
    @param [in] node - graph node to get the dependencies from.
    @param [out] pDependencies - pointer to to return the dependencies.
    @param [out] pNumDependencies -  returns the number of graph node dependencies.
    @returns #hipSuccess, #hipErrorInvalidValue
    pDependencies may be NULL, in which case this function will return the number of dependencies in
    pNumDependencies. Otherwise, pNumDependencies entries will be filled in. If pNumDependencies is
    higher than the actual number of dependencies, the remaining entries in pDependencies will be set
    to NULL, and the number of nodes actually obtained will be returned in pNumDependencies.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphNodeGetDependentNodes(node,pDependentNodes,pNumDependentNodes) nogil:
    """@brief Returns a node's dependent nodes.
    @param [in] node - graph node to get the Dependent nodes from.
    @param [out] pDependentNodes - pointer to return the graph dependent nodes.
    @param [out] pNumDependentNodes - returns the number of graph node dependent nodes.
    @returns #hipSuccess, #hipErrorInvalidValue
    DependentNodes may be NULL, in which case this function will return the number of dependent nodes
    in pNumDependentNodes. Otherwise, pNumDependentNodes entries will be filled in. If
    pNumDependentNodes is higher than the actual number of dependent nodes, the remaining entries in
    pDependentNodes will be set to NULL, and the number of nodes actually obtained will be returned
    in pNumDependentNodes.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphNodeGetType(node,pType) nogil:
    """@brief Returns a node's type.
    @param [in] node - instance of the graph to add dependencies.
    @param [out] pType - pointer to the return the type
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphDestroyNode(node) nogil:
    """@brief Remove a node from the graph.
    @param [in] node - graph node to remove
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphClone(pGraphClone,originalGraph) nogil:
    """@brief Clones a graph.
    @param [out] pGraphClone - Returns newly created cloned graph.
    @param [in] originalGraph - original graph to clone from.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphNodeFindInClone(pNode,originalNode,clonedGraph) nogil:
    """@brief Finds a cloned version of a node.
    @param [out] pNode - Returns the cloned node.
    @param [in] originalNode - original node handle.
    @param [in] clonedGraph - Cloned graph to query.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphInstantiate(pGraphExec,graph,pErrorNode,pLogBuffer,bufferSize) nogil:
    """@brief Creates an executable graph from a graph
    @param [out] pGraphExec - pointer to instantiated executable graph that is created.
    @param [in] graph - instance of graph to instantiate.
    @param [out] pErrorNode - pointer to error node in case error occured in graph instantiation,
    it could modify the correponding node.
    @param [out] pLogBuffer - pointer to log buffer.
    @param [out] bufferSize - the size of log buffer.
    @returns #hipSuccess, #hipErrorOutOfMemory
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphInstantiateWithFlags(pGraphExec,graph,flags) nogil:
    """@brief Creates an executable graph from a graph.
    @param [out] pGraphExec - pointer to instantiated executable graph that is created.
    @param [in] graph - instance of graph to instantiate.
    @param [in] flags - Flags to control instantiation.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphLaunch(graphExec,stream) nogil:
    """@brief launches an executable graph in a stream
    @param [in] graphExec - instance of executable graph to launch.
    @param [in] stream - instance of stream in which to launch executable graph.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphUpload(graphExec,stream) nogil:
    """@brief uploads an executable graph in a stream
    @param [in] graphExec - instance of executable graph to launch.
    @param [in] stream - instance of stream in which to launch executable graph.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecDestroy(graphExec) nogil:
    """@brief Destroys an executable graph
    @param [in] pGraphExec - instance of executable graph to destry.
    @returns #hipSuccess.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecUpdate(hGraphExec,hGraph,hErrorNode_out,updateResult_out) nogil:
    """@brief Check whether an executable graph can be updated with a graph and perform the update if  *
    possible.
    @param [in] hGraphExec - instance of executable graph to update.
    @param [in] hGraph - graph that contains the updated parameters.
    @param [in] hErrorNode_out -  node which caused the permissibility check to forbid the update.
    @param [in] updateResult_out - Whether the graph update was permitted.
    @returns #hipSuccess, #hipErrorGraphExecUpdateFailure
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddKernelNode(pGraphNode,graph,pDependencies,numDependencies,pNodeParams) nogil:
    """@brief Creates a kernel execution node and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - pointer to the dependencies on the kernel execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] pNodeParams - pointer to the parameters to the kernel execution node on the GPU.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDeviceFunction
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeGetParams(node,pNodeParams) nogil:
    """@brief Gets kernel node's parameters.
    @param [in] node - instance of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeSetParams(node,pNodeParams) nogil:
    """@brief Sets a kernel node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecKernelNodeSetParams(hGraphExec,node,pNodeParams) nogil:
    """@brief Sets the parameters for a kernel node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the kernel node parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNode(pGraphNode,graph,pDependencies,numDependencies,pCopyParams) nogil:
    """@brief Creates a memcpy node and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] pCopyParams - const pointer to the parameters for the memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeGetParams(node,pNodeParams) nogil:
    """@brief Gets a memcpy node's parameters.
    @param [in] node - instance of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeSetParams(node,pNodeParams) nogil:
    """@brief Sets a memcpy node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeSetAttribute(hNode,attr,value) nogil:
    """@brief Sets a node attribute.
    @param [in] hNode - instance of the node to set parameters to.
    @param [in] attr - the attribute node is set to.
    @param [in] value - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeGetAttribute(hNode,attr,value) nogil:
    """@brief Gets a node attribute.
    @param [in] hNode - instance of the node to set parameters to.
    @param [in] attr - the attribute node is set to.
    @param [in] value - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemcpyNodeSetParams(hGraphExec,node,pNodeParams) nogil:
    """@brief Sets the parameters for a memcpy node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the kernel node parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNode1D(pGraphNode,graph,pDependencies,numDependencies,dst,src,count,kind) nogil:
    """@brief Creates a 1D memcpy node and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] src - pointer to memory address to the source.
    @param [in] count - the size of the memory to copy.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeSetParams1D(node,dst,src,count,kind) nogil:
    """@brief Sets a memcpy node's parameters to perform a 1-dimensional copy.
    @param [in] node - instance of the node to set parameters to.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] src - pointer to memory address to the source.
    @param [in] count - the size of the memory to copy.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemcpyNodeSetParams1D(hGraphExec,node,dst,src,count,kind) nogil:
    """@brief Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional
    copy.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] src - pointer to memory address to the source.
    @param [in] count - the size of the memory to copy.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNodeFromSymbol(pGraphNode,graph,pDependencies,numDependencies,dst,symbol,count,offset,kind) nogil:
    """@brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] symbol - Device symbol address.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeSetParamsFromSymbol(node,dst,symbol,count,offset,kind) nogil:
    """@brief Sets a memcpy node's parameters to copy from a symbol on the device.
    @param [in] node - instance of the node to set parameters to.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] symbol - Device symbol address.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec,node,dst,symbol,count,offset,kind) nogil:
    """@brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the
    device.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] symbol - Device symbol address.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNodeToSymbol(pGraphNode,graph,pDependencies,numDependencies,symbol,src,count,offset,kind) nogil:
    """@brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] symbol - Device symbol address.
    @param [in] src - pointer to memory address of the src.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeSetParamsToSymbol(node,symbol,src,count,offset,kind) nogil:
    """@brief Sets a memcpy node's parameters to copy to a symbol on the device.
    @param [in] node - instance of the node to set parameters to.
    @param [in] symbol - Device symbol address.
    @param [in] src - pointer to memory address of the src.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec,node,symbol,src,count,offset,kind) nogil:
    """@brief Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the
    device.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] symbol - Device symbol address.
    @param [in] src - pointer to memory address of the src.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemsetNode(pGraphNode,graph,pDependencies,numDependencies,pMemsetParams) nogil:
    """@brief Creates a memset node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create.
    @param [in] graph - instance of the graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] pMemsetParams - const pointer to the parameters for the memory set.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemsetNodeGetParams(node,pNodeParams) nogil:
    """@brief Gets a memset node's parameters.
    @param [in] node - instane of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemsetNodeSetParams(node,pNodeParams) nogil:
    """@brief Sets a memset node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemsetNodeSetParams(hGraphExec,node,pNodeParams) nogil:
    """@brief Sets the parameters for a memset node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddHostNode(pGraphNode,graph,pDependencies,numDependencies,pNodeParams) nogil:
    """@brief Creates a host execution node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create.
    @param [in] graph - instance of the graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] pNodeParams -pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphHostNodeGetParams(node,pNodeParams) nogil:
    """@brief Returns a host node's parameters.
    @param [in] node - instane of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphHostNodeSetParams(node,pNodeParams) nogil:
    """@brief Sets a host node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecHostNodeSetParams(hGraphExec,node,pNodeParams) nogil:
    """@brief Sets the parameters for a host node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddChildGraphNode(pGraphNode,graph,pDependencies,numDependencies,childGraph) nogil:
    """@brief Creates a child graph node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create.
    @param [in] graph - instance of the graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] childGraph - the graph to clone into this node
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphChildGraphNodeGetGraph(node,pGraph) nogil:
    """@brief Gets a handle to the embedded graph of a child graph node.
    @param [in] node - instane of the node to get child graph.
    @param [out] pGraph - pointer to get the graph.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecChildGraphNodeSetParams(hGraphExec,node,childGraph) nogil:
    """@brief Updates node parameters in the child graph node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - node from the graph which was used to instantiate graphExec.
    @param [in] childGraph - child graph with updated parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddEmptyNode(pGraphNode,graph,pDependencies,numDependencies) nogil:
    """@brief Creates an empty node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    @param [in] graph - instane of the graph the node is add to.
    @param [in] pDependencies - const pointer to the node dependenties.
    @param [in] numDependencies - the number of dependencies.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddEventRecordNode(pGraphNode,graph,pDependencies,numDependencies,event) nogil:
    """@brief Creates an event record node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    @param [in] graph - instane of the graph the node to be added.
    @param [in] pDependencies - const pointer to the node dependenties.
    @param [in] numDependencies - the number of dependencies.
    @param [in] event - Event for the node.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphEventRecordNodeGetEvent(node,event_out) nogil:
    """@brief Returns the event associated with an event record node.
    @param [in] node -  instane of the node to get event from.
    @param [out] event_out - Pointer to return the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphEventRecordNodeSetEvent(node,event) nogil:
    """@brief Sets an event record node's event.
    @param [in] node - instane of the node to set event to.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecEventRecordNodeSetEvent(hGraphExec,hNode,event) nogil:
    """@brief Sets the event for an event record node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] hNode - node from the graph which was used to instantiate graphExec.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddEventWaitNode(pGraphNode,graph,pDependencies,numDependencies,event) nogil:
    """@brief Creates an event wait node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    @param [in] graph - instane of the graph the node to be added.
    @param [in] pDependencies - const pointer to the node dependenties.
    @param [in] numDependencies - the number of dependencies.
    @param [in] event - Event for the node.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphEventWaitNodeGetEvent(node,event_out) nogil:
    """@brief Returns the event associated with an event wait node.
    @param [in] node -  instane of the node to get event from.
    @param [out] event_out - Pointer to return the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphEventWaitNodeSetEvent(node,event) nogil:
    """@brief Sets an event wait node's event.
    @param [in] node - instane of the node to set event to.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecEventWaitNodeSetEvent(hGraphExec,hNode,event) nogil:
    """@brief Sets the event for an event record node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] hNode - node from the graph which was used to instantiate graphExec.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceGetGraphMemAttribute(device,attr,value) nogil:
    """@brief Get the mem attribute for graphs.
    @param [in] device - device the attr is get for.
    @param [in] attr - attr to get.
    @param [out] value - value for specific attr.
    @returns #hipSuccess, #hipErrorInvalidDevice
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceSetGraphMemAttribute(device,attr,value) nogil:
    """@brief Set the mem attribute for graphs.
    @param [in] device - device the attr is set for.
    @param [in] attr - attr to set.
    @param [in] value - value for specific attr.
    @returns #hipSuccess, #hipErrorInvalidDevice
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceGraphMemTrim(device) nogil:
    """@brief Free unused memory on specific device used for graph back to OS.
    @param [in] device - device the memory is used for graphs
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipUserObjectCreate(object_out,ptr,destroy,initialRefcount,flags) nogil:
    """@brief Create an instance of userObject to manage lifetime of a resource.
    @param [out] object_out - pointer to instace of userobj.
    @param [in] ptr - pointer to pass to destroy function.
    @param [in] destroy - destroy callback to remove resource.
    @param [in] initialRefcount - reference to resource.
    @param [in] flags - flags passed to API.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipUserObjectRelease(object,count) nogil:
    """@brief Release number of references to resource.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipUserObjectRetain(object,count) nogil:
    """@brief Retain number of references to resource.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphRetainUserObject(graph,object,count,flags) nogil:
    """@brief Retain user object for graphs.
    @param [in] graph - pointer to graph to retain the user object for.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @param [in] flags - flags passed to API.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphReleaseUserObject(graph,object,count) nogil:
    """@brief Release user object from graphs.
    @param [in] graph - pointer to graph to retain the user object for.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemAddressFree(devPtr,size) nogil:
    """@brief Frees an address range reservation made via hipMemAddressReserve
    @param [in] devPtr - starting address of the range.
    @param [in] size - size of the range.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemAddressReserve(ptr,size,alignment,addr,flags) nogil:
    """@brief Reserves an address range
    @param [out] ptr - starting address of the reserved range.
    @param [in] size - size of the reservation.
    @param [in] alignment - alignment of the address.
    @param [in] addr - requested starting address of the range.
    @param [in] flags - currently unused, must be zero.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemCreate(handle,size,prop,flags) nogil:
    """@brief Creates a memory allocation described by the properties and size
    @param [out] handle - value of the returned handle.
    @param [in] size - size of the allocation.
    @param [in] prop - properties of the allocation.
    @param [in] flags - currently unused, must be zero.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemExportToShareableHandle(shareableHandle,handle,handleType,flags) nogil:
    """@brief Exports an allocation to a requested shareable handle type.
    @param [out] shareableHandle - value of the returned handle.
    @param [in] handle - handle to share.
    @param [in] handleType - type of the shareable handle.
    @param [in] flags - currently unused, must be zero.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemGetAccess(flags,location,ptr) nogil:
    """@brief Get the access flags set for the given location and ptr.
    @param [out] flags - flags for this location.
    @param [in] location - target location.
    @param [in] ptr - address to check the access flags.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemGetAllocationGranularity(granularity,prop,option) nogil:
    """@brief Calculates either the minimal or recommended granularity.
    @param [out] granularity - returned granularity.
    @param [in] prop - location properties.
    @param [in] option - determines which granularity to return.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemGetAllocationPropertiesFromHandle(prop,handle) nogil:
    """@brief Retrieve the property structure of the given handle.
    @param [out] prop - properties of the given handle.
    @param [in] handle - handle to perform the query on.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemImportFromShareableHandle(handle,osHandle,shHandleType) nogil:
    """@brief Imports an allocation from a requested shareable handle type.
    @param [out] handle - returned value.
    @param [in] osHandle - shareable handle representing the memory allocation.
    @param [in] shHandleType - handle type.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemMap(ptr,size,offset,handle,flags) nogil:
    """@brief Maps an allocation handle to a reserved virtual address range.
    @param [in] ptr - address where the memory will be mapped.
    @param [in] size - size of the mapping.
    @param [in] offset - offset into the memory, currently must be zero.
    @param [in] handle - memory allocation to be mapped.
    @param [in] flags - currently unused, must be zero.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemMapArrayAsync(mapInfoList,count,stream) nogil:
    """@brief Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.
    @param [in] mapInfoList - list of hipArrayMapInfo.
    @param [in] count - number of hipArrayMapInfo in mapInfoList.
    @param [in] stream - stream identifier for the stream to use for map or unmap operations.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemRelease(handle) nogil:
    """@brief Release a memory handle representing a memory allocation which was previously allocated through hipMemCreate.
    @param [in] handle - handle of the memory allocation.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemRetainAllocationHandle(handle,addr) nogil:
    """@brief Returns the allocation handle of the backing memory allocation given the address.
    @param [out] handle - handle representing addr.
    @param [in] addr - address to look up.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemSetAccess(ptr,size,desc,count) nogil:
    """@brief Set the access flags for each location specified in desc for the given virtual address range.
    @param [in] ptr - starting address of the virtual address range.
    @param [in] size - size of the range.
    @param [in] desc - array of hipMemAccessDesc.
    @param [in] count - number of hipMemAccessDesc in desc.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemUnmap(ptr,size) nogil:
    """@brief Unmap memory allocation of a given address range.
    @param [in] ptr - starting address of the range to unmap.
    @param [in] size - size of the virtual address range.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGLGetDevices(pHipDeviceCount,pHipDevices,hipDeviceCount,deviceList) nogil:
    """
    """
    pass

def hipGraphicsGLRegisterBuffer(resource,buffer,flags) nogil:
    """
    """
    pass

def hipGraphicsGLRegisterImage(resource,image,target,flags) nogil:
    """
    """
    pass

def hipGraphicsMapResources(count,resources,stream) nogil:
    """
    """
    pass

def hipGraphicsSubResourceGetMappedArray(array,resource,arrayIndex,mipLevel) nogil:
    """
    """
    pass

def hipGraphicsResourceGetMappedPointer(devPtr,size,resource) nogil:
    """
    """
    pass

def hipGraphicsUnmapResources(count,resources,stream) nogil:
    """
    """
    pass

def hipGraphicsUnregisterResource(resource) nogil:
    """
    """
    pass

def hipMemcpy_spt(dst,src,sizeBytes,kind) nogil:
    """
    """
    pass

def hipMemcpyToSymbol_spt(symbol,src,sizeBytes,offset,kind) nogil:
    """
    """
    pass

def hipMemcpyFromSymbol_spt(dst,symbol,sizeBytes,offset,kind) nogil:
    """
    """
    pass

def hipMemcpy2D_spt(dst,dpitch,src,spitch,width,height,kind) nogil:
    """
    """
    pass

def hipMemcpy2DFromArray_spt(dst,dpitch,src,wOffset,hOffset,width,height,kind) nogil:
    """
    """
    pass

def hipMemcpy3D_spt(p) nogil:
    """
    """
    pass

def hipMemset_spt(dst,value,sizeBytes) nogil:
    """
    """
    pass

def hipMemsetAsync_spt(dst,value,sizeBytes,stream) nogil:
    """
    """
    pass

def hipMemset2D_spt(dst,pitch,value,width,height) nogil:
    """
    """
    pass

def hipMemset2DAsync_spt(dst,pitch,value,width,height,stream) nogil:
    """
    """
    pass

def hipMemset3DAsync_spt(pitchedDevPtr,value,extent,stream) nogil:
    """
    """
    pass

def hipMemset3D_spt(pitchedDevPtr,value,extent) nogil:
    """
    """
    pass

def hipMemcpyAsync_spt(dst,src,sizeBytes,kind,stream) nogil:
    """
    """
    pass

def hipMemcpy3DAsync_spt(p,stream) nogil:
    """
    """
    pass

def hipMemcpy2DAsync_spt(dst,dpitch,src,spitch,width,height,kind,stream) nogil:
    """
    """
    pass

def hipMemcpyFromSymbolAsync_spt(dst,symbol,sizeBytes,offset,kind,stream) nogil:
    """
    """
    pass

def hipMemcpyToSymbolAsync_spt(symbol,src,sizeBytes,offset,kind,stream) nogil:
    """
    """
    pass

def hipMemcpyFromArray_spt(dst,src,wOffsetSrc,hOffset,count,kind) nogil:
    """
    """
    pass

def hipMemcpy2DToArray_spt(dst,wOffset,hOffset,src,spitch,width,height,kind) nogil:
    """
    """
    pass

def hipMemcpy2DFromArrayAsync_spt(dst,dpitch,src,wOffsetSrc,hOffsetSrc,width,height,kind,stream) nogil:
    """
    """
    pass

def hipMemcpy2DToArrayAsync_spt(dst,wOffset,hOffset,src,spitch,width,height,kind,stream) nogil:
    """
    """
    pass

def hipStreamQuery_spt(stream) nogil:
    """
    """
    pass

def hipStreamSynchronize_spt(stream) nogil:
    """
    """
    pass

def hipStreamGetPriority_spt(stream,priority) nogil:
    """
    """
    pass

def hipStreamWaitEvent_spt(stream,event,flags) nogil:
    """
    """
    pass

def hipStreamGetFlags_spt(stream,flags) nogil:
    """
    """
    pass

def hipStreamAddCallback_spt(stream,callback,userData,flags) nogil:
    """
    """
    pass

def hipEventRecord_spt(event,stream) nogil:
    """
    """
    pass

def hipLaunchCooperativeKernel_spt(f,gridDim,blockDim,kernelParams,sharedMemBytes,hStream) nogil:
    """
    """
    pass

def hipLaunchKernel_spt(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream) nogil:
    """
    """
    pass

def hipGraphLaunch_spt(graphExec,stream) nogil:
    """
    """
    pass

def hipStreamBeginCapture_spt(stream,mode) nogil:
    """
    """
    pass

def hipStreamEndCapture_spt(stream,pGraph) nogil:
    """
    """
    pass

def hipStreamIsCapturing_spt(stream,pCaptureStatus) nogil:
    """
    """
    pass

def hipStreamGetCaptureInfo_spt(stream,pCaptureStatus,pId) nogil:
    """
    """
    pass

def hipStreamGetCaptureInfo_v2_spt(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out) nogil:
    """
    """
    pass

def hipLaunchHostFunc_spt(stream,fn,userData) nogil:
    """
    """
    pass

from chip cimport HIP_VERSION_MAJOR

from chip cimport HIP_VERSION_MINOR

from chip cimport HIP_VERSION_PATCH

from chip cimport HIP_VERSION_GITHASH

from chip cimport HIP_VERSION_BUILD_ID

from chip cimport HIP_VERSION_BUILD_NAME

from chip cimport HIP_VERSION

from chip cimport HIP_TRSA_OVERRIDE_FORMAT

from chip cimport HIP_TRSF_READ_AS_INTEGER

from chip cimport HIP_TRSF_NORMALIZED_COORDINATES

from chip cimport HIP_TRSF_SRGB

from chip cimport hipTextureType1D

from chip cimport hipTextureType2D

from chip cimport hipTextureType3D

from chip cimport hipTextureTypeCubemap

from chip cimport hipTextureType1DLayered

from chip cimport hipTextureType2DLayered

from chip cimport hipTextureTypeCubemapLayered

from chip cimport HIP_IMAGE_OBJECT_SIZE_DWORD

from chip cimport HIP_SAMPLER_OBJECT_SIZE_DWORD

from chip cimport HIP_SAMPLER_OBJECT_OFFSET_DWORD

from chip cimport HIP_TEXTURE_OBJECT_SIZE_DWORD

from chip cimport hipIpcMemLazyEnablePeerAccess

from chip cimport HIP_IPC_HANDLE_SIZE

from chip cimport hipStreamDefault

from chip cimport hipStreamNonBlocking

from chip cimport hipEventDefault

from chip cimport hipEventBlockingSync

from chip cimport hipEventDisableTiming

from chip cimport hipEventInterprocess

from chip cimport hipEventReleaseToDevice

from chip cimport hipEventReleaseToSystem

from chip cimport hipHostMallocDefault

from chip cimport hipHostMallocPortable

from chip cimport hipHostMallocMapped

from chip cimport hipHostMallocWriteCombined

from chip cimport hipHostMallocNumaUser

from chip cimport hipHostMallocCoherent

from chip cimport hipHostMallocNonCoherent

from chip cimport hipMemAttachGlobal

from chip cimport hipMemAttachHost

from chip cimport hipMemAttachSingle

from chip cimport hipDeviceMallocDefault

from chip cimport hipDeviceMallocFinegrained

from chip cimport hipMallocSignalMemory

from chip cimport hipHostRegisterDefault

from chip cimport hipHostRegisterPortable

from chip cimport hipHostRegisterMapped

from chip cimport hipHostRegisterIoMemory

from chip cimport hipExtHostRegisterCoarseGrained

from chip cimport hipDeviceScheduleAuto

from chip cimport hipDeviceScheduleSpin

from chip cimport hipDeviceScheduleYield

from chip cimport hipDeviceScheduleBlockingSync

from chip cimport hipDeviceScheduleMask

from chip cimport hipDeviceMapHost

from chip cimport hipDeviceLmemResizeToMax

from chip cimport hipArrayDefault

from chip cimport hipArrayLayered

from chip cimport hipArraySurfaceLoadStore

from chip cimport hipArrayCubemap

from chip cimport hipArrayTextureGather

from chip cimport hipOccupancyDefault

from chip cimport hipCooperativeLaunchMultiDeviceNoPreSync

from chip cimport hipCooperativeLaunchMultiDeviceNoPostSync

from chip cimport hipCpuDeviceId

from chip cimport hipInvalidDeviceId

from chip cimport hipExtAnyOrderLaunch

from chip cimport hipStreamWaitValueGte

from chip cimport hipStreamWaitValueEq

from chip cimport hipStreamWaitValueAnd

from chip cimport hipStreamWaitValueNor

from chip cimport hipStreamPerThread

class hipMemoryType(enum.IntEnum):
    hipMemoryTypeHost = 0
    hipMemoryTypeDevice = 1
    hipMemoryTypeArray = 2
    hipMemoryTypeUnified = 3
    hipMemoryTypeManaged = 4

class hipError_t(enum.IntEnum):
    hipSuccess = 0
    hipErrorInvalidValue = 1
    hipErrorOutOfMemory = 2
    hipErrorMemoryAllocation = 2
    hipErrorNotInitialized = 3
    hipErrorInitializationError = 3
    hipErrorDeinitialized = 4
    hipErrorProfilerDisabled = 5
    hipErrorProfilerNotInitialized = 6
    hipErrorProfilerAlreadyStarted = 7
    hipErrorProfilerAlreadyStopped = 8
    hipErrorInvalidConfiguration = 9
    hipErrorInvalidPitchValue = 12
    hipErrorInvalidSymbol = 13
    hipErrorInvalidDevicePointer = 17
    hipErrorInvalidMemcpyDirection = 21
    hipErrorInsufficientDriver = 35
    hipErrorMissingConfiguration = 52
    hipErrorPriorLaunchFailure = 53
    hipErrorInvalidDeviceFunction = 98
    hipErrorNoDevice = 100
    hipErrorInvalidDevice = 101
    hipErrorInvalidImage = 200
    hipErrorInvalidContext = 201
    hipErrorContextAlreadyCurrent = 202
    hipErrorMapFailed = 205
    hipErrorMapBufferObjectFailed = 205
    hipErrorUnmapFailed = 206
    hipErrorArrayIsMapped = 207
    hipErrorAlreadyMapped = 208
    hipErrorNoBinaryForGpu = 209
    hipErrorAlreadyAcquired = 210
    hipErrorNotMapped = 211
    hipErrorNotMappedAsArray = 212
    hipErrorNotMappedAsPointer = 213
    hipErrorECCNotCorrectable = 214
    hipErrorUnsupportedLimit = 215
    hipErrorContextAlreadyInUse = 216
    hipErrorPeerAccessUnsupported = 217
    hipErrorInvalidKernelFile = 218
    hipErrorInvalidGraphicsContext = 219
    hipErrorInvalidSource = 300
    hipErrorFileNotFound = 301
    hipErrorSharedObjectSymbolNotFound = 302
    hipErrorSharedObjectInitFailed = 303
    hipErrorOperatingSystem = 304
    hipErrorInvalidHandle = 400
    hipErrorInvalidResourceHandle = 400
    hipErrorIllegalState = 401
    hipErrorNotFound = 500
    hipErrorNotReady = 600
    hipErrorIllegalAddress = 700
    hipErrorLaunchOutOfResources = 701
    hipErrorLaunchTimeOut = 702
    hipErrorPeerAccessAlreadyEnabled = 704
    hipErrorPeerAccessNotEnabled = 705
    hipErrorSetOnActiveProcess = 708
    hipErrorContextIsDestroyed = 709
    hipErrorAssert = 710
    hipErrorHostMemoryAlreadyRegistered = 712
    hipErrorHostMemoryNotRegistered = 713
    hipErrorLaunchFailure = 719
    hipErrorCooperativeLaunchTooLarge = 720
    hipErrorNotSupported = 801
    hipErrorStreamCaptureUnsupported = 900
    hipErrorStreamCaptureInvalidated = 901
    hipErrorStreamCaptureMerge = 902
    hipErrorStreamCaptureUnmatched = 903
    hipErrorStreamCaptureUnjoined = 904
    hipErrorStreamCaptureIsolation = 905
    hipErrorStreamCaptureImplicit = 906
    hipErrorCapturedEvent = 907
    hipErrorStreamCaptureWrongThread = 908
    hipErrorGraphExecUpdateFailure = 910
    hipErrorUnknown = 999
    hipErrorRuntimeMemory = 1052
    hipErrorRuntimeOther = 1053
    hipErrorTbd = 1054

class hipDeviceAttribute_t(enum.IntEnum):
    hipDeviceAttributeCudaCompatibleBegin = 0
    hipDeviceAttributeEccEnabled = 0
    hipDeviceAttributeAccessPolicyMaxWindowSize = 1
    hipDeviceAttributeAsyncEngineCount = 2
    hipDeviceAttributeCanMapHostMemory = 3
    hipDeviceAttributeCanUseHostPointerForRegisteredMem = 4
    hipDeviceAttributeClockRate = 5
    hipDeviceAttributeComputeMode = 6
    hipDeviceAttributeComputePreemptionSupported = 7
    hipDeviceAttributeConcurrentKernels = 8
    hipDeviceAttributeConcurrentManagedAccess = 9
    hipDeviceAttributeCooperativeLaunch = 10
    hipDeviceAttributeCooperativeMultiDeviceLaunch = 11
    hipDeviceAttributeDeviceOverlap = 12
    hipDeviceAttributeDirectManagedMemAccessFromHost = 13
    hipDeviceAttributeGlobalL1CacheSupported = 14
    hipDeviceAttributeHostNativeAtomicSupported = 15
    hipDeviceAttributeIntegrated = 16
    hipDeviceAttributeIsMultiGpuBoard = 17
    hipDeviceAttributeKernelExecTimeout = 18
    hipDeviceAttributeL2CacheSize = 19
    hipDeviceAttributeLocalL1CacheSupported = 20
    hipDeviceAttributeLuid = 21
    hipDeviceAttributeLuidDeviceNodeMask = 22
    hipDeviceAttributeComputeCapabilityMajor = 23
    hipDeviceAttributeManagedMemory = 24
    hipDeviceAttributeMaxBlocksPerMultiProcessor = 25
    hipDeviceAttributeMaxBlockDimX = 26
    hipDeviceAttributeMaxBlockDimY = 27
    hipDeviceAttributeMaxBlockDimZ = 28
    hipDeviceAttributeMaxGridDimX = 29
    hipDeviceAttributeMaxGridDimY = 30
    hipDeviceAttributeMaxGridDimZ = 31
    hipDeviceAttributeMaxSurface1D = 32
    hipDeviceAttributeMaxSurface1DLayered = 33
    hipDeviceAttributeMaxSurface2D = 34
    hipDeviceAttributeMaxSurface2DLayered = 35
    hipDeviceAttributeMaxSurface3D = 36
    hipDeviceAttributeMaxSurfaceCubemap = 37
    hipDeviceAttributeMaxSurfaceCubemapLayered = 38
    hipDeviceAttributeMaxTexture1DWidth = 39
    hipDeviceAttributeMaxTexture1DLayered = 40
    hipDeviceAttributeMaxTexture1DLinear = 41
    hipDeviceAttributeMaxTexture1DMipmap = 42
    hipDeviceAttributeMaxTexture2DWidth = 43
    hipDeviceAttributeMaxTexture2DHeight = 44
    hipDeviceAttributeMaxTexture2DGather = 45
    hipDeviceAttributeMaxTexture2DLayered = 46
    hipDeviceAttributeMaxTexture2DLinear = 47
    hipDeviceAttributeMaxTexture2DMipmap = 48
    hipDeviceAttributeMaxTexture3DWidth = 49
    hipDeviceAttributeMaxTexture3DHeight = 50
    hipDeviceAttributeMaxTexture3DDepth = 51
    hipDeviceAttributeMaxTexture3DAlt = 52
    hipDeviceAttributeMaxTextureCubemap = 53
    hipDeviceAttributeMaxTextureCubemapLayered = 54
    hipDeviceAttributeMaxThreadsDim = 55
    hipDeviceAttributeMaxThreadsPerBlock = 56
    hipDeviceAttributeMaxThreadsPerMultiProcessor = 57
    hipDeviceAttributeMaxPitch = 58
    hipDeviceAttributeMemoryBusWidth = 59
    hipDeviceAttributeMemoryClockRate = 60
    hipDeviceAttributeComputeCapabilityMinor = 61
    hipDeviceAttributeMultiGpuBoardGroupID = 62
    hipDeviceAttributeMultiprocessorCount = 63
    hipDeviceAttributeName = 64
    hipDeviceAttributePageableMemoryAccess = 65
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 66
    hipDeviceAttributePciBusId = 67
    hipDeviceAttributePciDeviceId = 68
    hipDeviceAttributePciDomainID = 69
    hipDeviceAttributePersistingL2CacheMaxSize = 70
    hipDeviceAttributeMaxRegistersPerBlock = 71
    hipDeviceAttributeMaxRegistersPerMultiprocessor = 72
    hipDeviceAttributeReservedSharedMemPerBlock = 73
    hipDeviceAttributeMaxSharedMemoryPerBlock = 74
    hipDeviceAttributeSharedMemPerBlockOptin = 75
    hipDeviceAttributeSharedMemPerMultiprocessor = 76
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio = 77
    hipDeviceAttributeStreamPrioritiesSupported = 78
    hipDeviceAttributeSurfaceAlignment = 79
    hipDeviceAttributeTccDriver = 80
    hipDeviceAttributeTextureAlignment = 81
    hipDeviceAttributeTexturePitchAlignment = 82
    hipDeviceAttributeTotalConstantMemory = 83
    hipDeviceAttributeTotalGlobalMem = 84
    hipDeviceAttributeUnifiedAddressing = 85
    hipDeviceAttributeUuid = 86
    hipDeviceAttributeWarpSize = 87
    hipDeviceAttributeMemoryPoolsSupported = 88
    hipDeviceAttributeVirtualMemoryManagementSupported = 89
    hipDeviceAttributeCudaCompatibleEnd = 9999
    hipDeviceAttributeAmdSpecificBegin = 10000
    hipDeviceAttributeClockInstructionRate = 10000
    hipDeviceAttributeArch = 10001
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 10002
    hipDeviceAttributeGcnArch = 10003
    hipDeviceAttributeGcnArchName = 10004
    hipDeviceAttributeHdpMemFlushCntl = 10005
    hipDeviceAttributeHdpRegFlushCntl = 10006
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = 10007
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = 10008
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = 10009
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = 10010
    hipDeviceAttributeIsLargeBar = 10011
    hipDeviceAttributeAsicRevision = 10012
    hipDeviceAttributeCanUseStreamWaitValue = 10013
    hipDeviceAttributeImageSupport = 10014
    hipDeviceAttributePhysicalMultiProcessorCount = 10015
    hipDeviceAttributeFineGrainSupport = 10016
    hipDeviceAttributeWallClockRate = 10017
    hipDeviceAttributeAmdSpecificEnd = 19999
    hipDeviceAttributeVendorSpecificBegin = 20000

class hipComputeMode(enum.IntEnum):
    hipComputeModeDefault = 0
    hipComputeModeExclusive = 1
    hipComputeModeProhibited = 2
    hipComputeModeExclusiveProcess = 3

class hipChannelFormatKind(enum.IntEnum):
    hipChannelFormatKindSigned = 0
    hipChannelFormatKindUnsigned = 1
    hipChannelFormatKindFloat = 2
    hipChannelFormatKindNone = 3

class hipArray_Format(enum.IntEnum):
    HIP_AD_FORMAT_UNSIGNED_INT8 = 1
    HIP_AD_FORMAT_UNSIGNED_INT16 = 2
    HIP_AD_FORMAT_UNSIGNED_INT32 = 3
    HIP_AD_FORMAT_SIGNED_INT8 = 8
    HIP_AD_FORMAT_SIGNED_INT16 = 9
    HIP_AD_FORMAT_SIGNED_INT32 = 10
    HIP_AD_FORMAT_HALF = 16
    HIP_AD_FORMAT_FLOAT = 32

class hipResourceType(enum.IntEnum):
    hipResourceTypeArray = 0
    hipResourceTypeMipmappedArray = 1
    hipResourceTypeLinear = 2
    hipResourceTypePitch2D = 3

class HIPresourcetype_enum(enum.IntEnum):
    HIP_RESOURCE_TYPE_ARRAY = 0
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    HIP_RESOURCE_TYPE_LINEAR = 2
    HIP_RESOURCE_TYPE_PITCH2D = 3

class HIPresourcetype(enum.IntEnum):
    HIP_RESOURCE_TYPE_ARRAY = 0
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    HIP_RESOURCE_TYPE_LINEAR = 2
    HIP_RESOURCE_TYPE_PITCH2D = 3

class hipResourcetype(enum.IntEnum):
    HIP_RESOURCE_TYPE_ARRAY = 0
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    HIP_RESOURCE_TYPE_LINEAR = 2
    HIP_RESOURCE_TYPE_PITCH2D = 3

class HIPaddress_mode_enum(enum.IntEnum):
    HIP_TR_ADDRESS_MODE_WRAP = 0
    HIP_TR_ADDRESS_MODE_CLAMP = 1
    HIP_TR_ADDRESS_MODE_MIRROR = 2
    HIP_TR_ADDRESS_MODE_BORDER = 3

class HIPaddress_mode(enum.IntEnum):
    HIP_TR_ADDRESS_MODE_WRAP = 0
    HIP_TR_ADDRESS_MODE_CLAMP = 1
    HIP_TR_ADDRESS_MODE_MIRROR = 2
    HIP_TR_ADDRESS_MODE_BORDER = 3

class HIPfilter_mode_enum(enum.IntEnum):
    HIP_TR_FILTER_MODE_POINT = 0
    HIP_TR_FILTER_MODE_LINEAR = 1

class HIPfilter_mode(enum.IntEnum):
    HIP_TR_FILTER_MODE_POINT = 0
    HIP_TR_FILTER_MODE_LINEAR = 1

class hipResourceViewFormat(enum.IntEnum):
    hipResViewFormatNone = 0
    hipResViewFormatUnsignedChar1 = 1
    hipResViewFormatUnsignedChar2 = 2
    hipResViewFormatUnsignedChar4 = 3
    hipResViewFormatSignedChar1 = 4
    hipResViewFormatSignedChar2 = 5
    hipResViewFormatSignedChar4 = 6
    hipResViewFormatUnsignedShort1 = 7
    hipResViewFormatUnsignedShort2 = 8
    hipResViewFormatUnsignedShort4 = 9
    hipResViewFormatSignedShort1 = 10
    hipResViewFormatSignedShort2 = 11
    hipResViewFormatSignedShort4 = 12
    hipResViewFormatUnsignedInt1 = 13
    hipResViewFormatUnsignedInt2 = 14
    hipResViewFormatUnsignedInt4 = 15
    hipResViewFormatSignedInt1 = 16
    hipResViewFormatSignedInt2 = 17
    hipResViewFormatSignedInt4 = 18
    hipResViewFormatHalf1 = 19
    hipResViewFormatHalf2 = 20
    hipResViewFormatHalf4 = 21
    hipResViewFormatFloat1 = 22
    hipResViewFormatFloat2 = 23
    hipResViewFormatFloat4 = 24
    hipResViewFormatUnsignedBlockCompressed1 = 25
    hipResViewFormatUnsignedBlockCompressed2 = 26
    hipResViewFormatUnsignedBlockCompressed3 = 27
    hipResViewFormatUnsignedBlockCompressed4 = 28
    hipResViewFormatSignedBlockCompressed4 = 29
    hipResViewFormatUnsignedBlockCompressed5 = 30
    hipResViewFormatSignedBlockCompressed5 = 31
    hipResViewFormatUnsignedBlockCompressed6H = 32
    hipResViewFormatSignedBlockCompressed6H = 33
    hipResViewFormatUnsignedBlockCompressed7 = 34

class HIPresourceViewFormat_enum(enum.IntEnum):
    HIP_RES_VIEW_FORMAT_NONE = 0
    HIP_RES_VIEW_FORMAT_UINT_1X8 = 1
    HIP_RES_VIEW_FORMAT_UINT_2X8 = 2
    HIP_RES_VIEW_FORMAT_UINT_4X8 = 3
    HIP_RES_VIEW_FORMAT_SINT_1X8 = 4
    HIP_RES_VIEW_FORMAT_SINT_2X8 = 5
    HIP_RES_VIEW_FORMAT_SINT_4X8 = 6
    HIP_RES_VIEW_FORMAT_UINT_1X16 = 7
    HIP_RES_VIEW_FORMAT_UINT_2X16 = 8
    HIP_RES_VIEW_FORMAT_UINT_4X16 = 9
    HIP_RES_VIEW_FORMAT_SINT_1X16 = 10
    HIP_RES_VIEW_FORMAT_SINT_2X16 = 11
    HIP_RES_VIEW_FORMAT_SINT_4X16 = 12
    HIP_RES_VIEW_FORMAT_UINT_1X32 = 13
    HIP_RES_VIEW_FORMAT_UINT_2X32 = 14
    HIP_RES_VIEW_FORMAT_UINT_4X32 = 15
    HIP_RES_VIEW_FORMAT_SINT_1X32 = 16
    HIP_RES_VIEW_FORMAT_SINT_2X32 = 17
    HIP_RES_VIEW_FORMAT_SINT_4X32 = 18
    HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 19
    HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 20
    HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 21
    HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 22
    HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 23
    HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 24
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
    HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 29
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
    HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 31
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 33
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34

class HIPresourceViewFormat(enum.IntEnum):
    HIP_RES_VIEW_FORMAT_NONE = 0
    HIP_RES_VIEW_FORMAT_UINT_1X8 = 1
    HIP_RES_VIEW_FORMAT_UINT_2X8 = 2
    HIP_RES_VIEW_FORMAT_UINT_4X8 = 3
    HIP_RES_VIEW_FORMAT_SINT_1X8 = 4
    HIP_RES_VIEW_FORMAT_SINT_2X8 = 5
    HIP_RES_VIEW_FORMAT_SINT_4X8 = 6
    HIP_RES_VIEW_FORMAT_UINT_1X16 = 7
    HIP_RES_VIEW_FORMAT_UINT_2X16 = 8
    HIP_RES_VIEW_FORMAT_UINT_4X16 = 9
    HIP_RES_VIEW_FORMAT_SINT_1X16 = 10
    HIP_RES_VIEW_FORMAT_SINT_2X16 = 11
    HIP_RES_VIEW_FORMAT_SINT_4X16 = 12
    HIP_RES_VIEW_FORMAT_UINT_1X32 = 13
    HIP_RES_VIEW_FORMAT_UINT_2X32 = 14
    HIP_RES_VIEW_FORMAT_UINT_4X32 = 15
    HIP_RES_VIEW_FORMAT_SINT_1X32 = 16
    HIP_RES_VIEW_FORMAT_SINT_2X32 = 17
    HIP_RES_VIEW_FORMAT_SINT_4X32 = 18
    HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 19
    HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 20
    HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 21
    HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 22
    HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 23
    HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 24
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
    HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 29
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
    HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 31
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 33
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34

class hipMemcpyKind(enum.IntEnum):
    hipMemcpyHostToHost = 0
    hipMemcpyHostToDevice = 1
    hipMemcpyDeviceToHost = 2
    hipMemcpyDeviceToDevice = 3
    hipMemcpyDefault = 4

class hipFunction_attribute(enum.IntEnum):
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
    HIP_FUNC_ATTRIBUTE_NUM_REGS = 4
    HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6
    HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    HIP_FUNC_ATTRIBUTE_MAX = 10

class hipPointer_attribute(enum.IntEnum):
    HIP_POINTER_ATTRIBUTE_CONTEXT = 1
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
    HIP_POINTER_ATTRIBUTE_HOST_POINTER = 4
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS = 5
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
    HIP_POINTER_ATTRIBUTE_BUFFER_ID = 7
    HIP_POINTER_ATTRIBUTE_IS_MANAGED = 8
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 10
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE = 12
    HIP_POINTER_ATTRIBUTE_MAPPED = 13
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17

def hipCreateChannelDesc(x,y,z,w,f) nogil:
    """
    """
    pass

class hipTextureAddressMode(enum.IntEnum):
    hipAddressModeWrap = 0
    hipAddressModeClamp = 1
    hipAddressModeMirror = 2
    hipAddressModeBorder = 3

class hipTextureFilterMode(enum.IntEnum):
    hipFilterModePoint = 0
    hipFilterModeLinear = 1

class hipTextureReadMode(enum.IntEnum):
    hipReadModeElementType = 0
    hipReadModeNormalizedFloat = 1

class hipSurfaceBoundaryMode(enum.IntEnum):
    hipBoundaryModeZero = 0
    hipBoundaryModeTrap = 1
    hipBoundaryModeClamp = 2

class hipDeviceP2PAttr(enum.IntEnum):
    hipDevP2PAttrPerformanceRank = 0
    hipDevP2PAttrAccessSupported = 1
    hipDevP2PAttrNativeAtomicSupported = 2
    hipDevP2PAttrHipArrayAccessSupported = 3

class hipLimit_t(enum.IntEnum):
    hipLimitStackSize = 0
    hipLimitPrintfFifoSize = 1
    hipLimitMallocHeapSize = 2
    hipLimitRange = 3

class hipMemoryAdvise(enum.IntEnum):
    hipMemAdviseSetReadMostly = 1
    hipMemAdviseUnsetReadMostly = 2
    hipMemAdviseSetPreferredLocation = 3
    hipMemAdviseUnsetPreferredLocation = 4
    hipMemAdviseSetAccessedBy = 5
    hipMemAdviseUnsetAccessedBy = 6
    hipMemAdviseSetCoarseGrain = 100
    hipMemAdviseUnsetCoarseGrain = 101

class hipMemRangeCoherencyMode(enum.IntEnum):
    hipMemRangeCoherencyModeFineGrain = 0
    hipMemRangeCoherencyModeCoarseGrain = 1
    hipMemRangeCoherencyModeIndeterminate = 2

class hipMemRangeAttribute(enum.IntEnum):
    hipMemRangeAttributeReadMostly = 1
    hipMemRangeAttributePreferredLocation = 2
    hipMemRangeAttributeAccessedBy = 3
    hipMemRangeAttributeLastPrefetchLocation = 4
    hipMemRangeAttributeCoherencyMode = 100

class hipMemPoolAttr(enum.IntEnum):
    hipMemPoolReuseFollowEventDependencies = 1
    hipMemPoolReuseAllowOpportunistic = 2
    hipMemPoolReuseAllowInternalDependencies = 3
    hipMemPoolAttrReleaseThreshold = 4
    hipMemPoolAttrReservedMemCurrent = 5
    hipMemPoolAttrReservedMemHigh = 6
    hipMemPoolAttrUsedMemCurrent = 7
    hipMemPoolAttrUsedMemHigh = 8

class hipMemLocationType(enum.IntEnum):
    hipMemLocationTypeInvalid = 0
    hipMemLocationTypeDevice = 1

class hipMemAccessFlags(enum.IntEnum):
    hipMemAccessFlagsProtNone = 0
    hipMemAccessFlagsProtRead = 1
    hipMemAccessFlagsProtReadWrite = 3

class hipMemAllocationType(enum.IntEnum):
    hipMemAllocationTypeInvalid = 0
    hipMemAllocationTypePinned = 1
    hipMemAllocationTypeMax = 2147483647

class hipMemAllocationHandleType(enum.IntEnum):
    hipMemHandleTypeNone = 0
    hipMemHandleTypePosixFileDescriptor = 1
    hipMemHandleTypeWin32 = 2
    hipMemHandleTypeWin32Kmt = 4

class hipJitOption(enum.IntEnum):
    hipJitOptionMaxRegisters = 0
    hipJitOptionThreadsPerBlock = 1
    hipJitOptionWallTime = 2
    hipJitOptionInfoLogBuffer = 3
    hipJitOptionInfoLogBufferSizeBytes = 4
    hipJitOptionErrorLogBuffer = 5
    hipJitOptionErrorLogBufferSizeBytes = 6
    hipJitOptionOptimizationLevel = 7
    hipJitOptionTargetFromContext = 8
    hipJitOptionTarget = 9
    hipJitOptionFallbackStrategy = 10
    hipJitOptionGenerateDebugInfo = 11
    hipJitOptionLogVerbose = 12
    hipJitOptionGenerateLineInfo = 13
    hipJitOptionCacheMode = 14
    hipJitOptionSm3xOpt = 15
    hipJitOptionFastCompile = 16
    hipJitOptionNumOptions = 17

class hipFuncAttribute(enum.IntEnum):
    hipFuncAttributeMaxDynamicSharedMemorySize = 8
    hipFuncAttributePreferredSharedMemoryCarveout = 9
    hipFuncAttributeMax = 10

class hipFuncCache_t(enum.IntEnum):
    hipFuncCachePreferNone = 0
    hipFuncCachePreferShared = 1
    hipFuncCachePreferL1 = 2
    hipFuncCachePreferEqual = 3

class hipSharedMemConfig(enum.IntEnum):
    hipSharedMemBankSizeDefault = 0
    hipSharedMemBankSizeFourByte = 1
    hipSharedMemBankSizeEightByte = 2

class hipExternalMemoryHandleType_enum(enum.IntEnum):
    hipExternalMemoryHandleTypeOpaqueFd = 1
    hipExternalMemoryHandleTypeOpaqueWin32 = 2
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3
    hipExternalMemoryHandleTypeD3D12Heap = 4
    hipExternalMemoryHandleTypeD3D12Resource = 5
    hipExternalMemoryHandleTypeD3D11Resource = 6
    hipExternalMemoryHandleTypeD3D11ResourceKmt = 7

class hipExternalMemoryHandleType(enum.IntEnum):
    hipExternalMemoryHandleTypeOpaqueFd = 1
    hipExternalMemoryHandleTypeOpaqueWin32 = 2
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3
    hipExternalMemoryHandleTypeD3D12Heap = 4
    hipExternalMemoryHandleTypeD3D12Resource = 5
    hipExternalMemoryHandleTypeD3D11Resource = 6
    hipExternalMemoryHandleTypeD3D11ResourceKmt = 7

class hipExternalSemaphoreHandleType_enum(enum.IntEnum):
    hipExternalSemaphoreHandleTypeOpaqueFd = 1
    hipExternalSemaphoreHandleTypeOpaqueWin32 = 2
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3
    hipExternalSemaphoreHandleTypeD3D12Fence = 4

class hipExternalSemaphoreHandleType(enum.IntEnum):
    hipExternalSemaphoreHandleTypeOpaqueFd = 1
    hipExternalSemaphoreHandleTypeOpaqueWin32 = 2
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3
    hipExternalSemaphoreHandleTypeD3D12Fence = 4

class hipGLDeviceList(enum.IntEnum):
    hipGLDeviceListAll = 1
    hipGLDeviceListCurrentFrame = 2
    hipGLDeviceListNextFrame = 3

class hipGraphicsRegisterFlags(enum.IntEnum):
    hipGraphicsRegisterFlagsNone = 0
    hipGraphicsRegisterFlagsReadOnly = 1
    hipGraphicsRegisterFlagsWriteDiscard = 2
    hipGraphicsRegisterFlagsSurfaceLoadStore = 4
    hipGraphicsRegisterFlagsTextureGather = 8

class hipGraphNodeType(enum.IntEnum):
    hipGraphNodeTypeKernel = 0
    hipGraphNodeTypeMemcpy = 1
    hipGraphNodeTypeMemset = 2
    hipGraphNodeTypeHost = 3
    hipGraphNodeTypeGraph = 4
    hipGraphNodeTypeEmpty = 5
    hipGraphNodeTypeWaitEvent = 6
    hipGraphNodeTypeEventRecord = 7
    hipGraphNodeTypeExtSemaphoreSignal = 8
    hipGraphNodeTypeExtSemaphoreWait = 9
    hipGraphNodeTypeMemcpyFromSymbol = 10
    hipGraphNodeTypeMemcpyToSymbol = 11
    hipGraphNodeTypeCount = 12

class hipKernelNodeAttrID(enum.IntEnum):
    hipKernelNodeAttributeAccessPolicyWindow = 1
    hipKernelNodeAttributeCooperative = 2

class hipAccessProperty(enum.IntEnum):
    hipAccessPropertyNormal = 0
    hipAccessPropertyStreaming = 1
    hipAccessPropertyPersisting = 2

class hipGraphExecUpdateResult(enum.IntEnum):
    hipGraphExecUpdateSuccess = 0
    hipGraphExecUpdateError = 1
    hipGraphExecUpdateErrorTopologyChanged = 2
    hipGraphExecUpdateErrorNodeTypeChanged = 3
    hipGraphExecUpdateErrorFunctionChanged = 4
    hipGraphExecUpdateErrorParametersChanged = 5
    hipGraphExecUpdateErrorNotSupported = 6
    hipGraphExecUpdateErrorUnsupportedFunctionChange = 7

class hipStreamCaptureMode(enum.IntEnum):
    hipStreamCaptureModeGlobal = 0
    hipStreamCaptureModeThreadLocal = 1
    hipStreamCaptureModeRelaxed = 2

class hipStreamCaptureStatus(enum.IntEnum):
    hipStreamCaptureStatusNone = 0
    hipStreamCaptureStatusActive = 1
    hipStreamCaptureStatusInvalidated = 2

class hipStreamUpdateCaptureDependenciesFlags(enum.IntEnum):
    hipStreamAddCaptureDependencies = 0
    hipStreamSetCaptureDependencies = 1

class hipGraphMemAttributeType(enum.IntEnum):
    hipGraphMemAttrUsedMemCurrent = 0
    hipGraphMemAttrUsedMemHigh = 1
    hipGraphMemAttrReservedMemCurrent = 2
    hipGraphMemAttrReservedMemHigh = 3

class hipUserObjectFlags(enum.IntEnum):
    hipUserObjectNoDestructorSync = 1

class hipUserObjectRetainFlags(enum.IntEnum):
    hipGraphUserObjectMove = 1

class hipGraphInstantiateFlags(enum.IntEnum):
    hipGraphInstantiateFlagAutoFreeOnLaunch = 1

class hipMemAllocationGranularity_flags(enum.IntEnum):
    hipMemAllocationGranularityMinimum = 0
    hipMemAllocationGranularityRecommended = 1

class hipMemHandleType(enum.IntEnum):
    hipMemHandleTypeGeneric = 0

class hipMemOperationType(enum.IntEnum):
    hipMemOperationTypeMap = 1
    hipMemOperationTypeUnmap = 2

class hipArraySparseSubresourceType(enum.IntEnum):
    hipArraySparseSubresourceTypeSparseLevel = 0
    hipArraySparseSubresourceTypeMiptail = 1

def hipInit(flags) nogil:
    """@defgroup API HIP API
    @{
    Defines the HIP API.  See the individual sections for more information.
    @defgroup Driver Initialization and Version
    @{
    This section describes the initializtion and version functions of HIP runtime API.
    @brief Explicitly initializes the HIP runtime.
    Most HIP APIs implicitly initialize the HIP runtime.
    This API provides control over the timing of the initialization.
    """
    pass

def hipDriverGetVersion(driverVersion) nogil:
    """@brief Returns the approximate HIP driver version.
    @param [out] driverVersion
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning The HIP feature set does not correspond to an exact CUDA SDK driver revision.
    This function always set *driverVersion to 4 as an approximation though HIP supports
    some features which were introduced in later CUDA SDK revisions.
    HIP apps code should not rely on the driver revision number here and should
    use arch feature flags to test device capabilities or conditional compilation.
    @see hipRuntimeGetVersion
    """
    pass

def hipRuntimeGetVersion(runtimeVersion) nogil:
    """@brief Returns the approximate HIP Runtime version.
    @param [out] runtimeVersion
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning The version definition of HIP runtime is different from CUDA.
    On AMD platform, the function returns HIP runtime version,
    while on NVIDIA platform, it returns CUDA runtime version.
    And there is no mapping/correlation between HIP version and CUDA version.
    @see hipDriverGetVersion
    """
    pass

def hipDeviceGet(device,ordinal) nogil:
    """@brief Returns a handle to a compute device
    @param [out] device
    @param [in] ordinal
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceComputeCapability(major,minor,device) nogil:
    """@brief Returns the compute capability of the device
    @param [out] major
    @param [out] minor
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceGetName(name,len,device) nogil:
    """@brief Returns an identifer string for the device.
    @param [out] name
    @param [in] len
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceGetUuid(uuid,device) nogil:
    """@brief Returns an UUID for the device.[BETA]
    @param [out] uuid
    @param [in] device
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotInitialized,
    #hipErrorDeinitialized
    """
    pass

def hipDeviceGetP2PAttribute(value,attr,srcDevice,dstDevice) nogil:
    """@brief Returns a value for attr of link between two devices
    @param [out] value
    @param [in] attr
    @param [in] srcDevice
    @param [in] dstDevice
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceGetPCIBusId(pciBusId,len,device) nogil:
    """@brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
    @param [out] pciBusId
    @param [in] len
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceGetByPCIBusId(device,pciBusId) nogil:
    """@brief Returns a handle to a compute device.
    @param [out] device handle
    @param [in] PCI Bus ID
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    """
    pass

def hipDeviceTotalMem(bytes,device) nogil:
    """@brief Returns the total amount of memory on the device.
    @param [out] bytes
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    pass

def hipDeviceSynchronize() nogil:
    """@}
    @defgroup Device Device Management
    @{
    This section describes the device management functions of HIP runtime API.
    @brief Waits on all active streams on current device
    When this command is invoked, the host thread gets blocked until all the commands associated
    with streams associated with the device. HIP does not support multiple blocking modes (yet!).
    @returns #hipSuccess
    @see hipSetDevice, hipDeviceReset
    """
    pass

def hipDeviceReset() nogil:
    """@brief The state of current device is discarded and updated to a fresh state.
    Calling this function deletes all streams created, memory allocated, kernels running, events
    created. Make sure that no other thread is using the device or streams, memory, kernels, events
    associated with the current device.
    @returns #hipSuccess
    @see hipDeviceSynchronize
    """
    pass

def hipSetDevice(deviceId) nogil:
    """@brief Set default device to be used for subsequent hip API calls from this thread.
    @param[in] deviceId Valid device in range 0...hipGetDeviceCount().
    Sets @p device as the default device for the calling host thread.  Valid device id's are 0...
    (hipGetDeviceCount()-1).
    Many HIP APIs implicitly use the "default device" :
    - Any device memory subsequently allocated from this host thread (using hipMalloc) will be
    allocated on device.
    - Any streams or events created from this host thread will be associated with device.
    - Any kernels launched from this host thread (using hipLaunchKernel) will be executed on device
    (unless a specific stream is specified, in which case the device associated with that stream will
    be used).
    This function may be called from any host thread.  Multiple host threads may use the same device.
    This function does no synchronization with the previous or new device, and has very little
    runtime overhead. Applications can use hipSetDevice to quickly switch the default device before
    making a HIP runtime call which uses the default device.
    The default device is stored in thread-local-storage for each thread.
    Thread-pool implementations may inherit the default device of the previous thread.  A good
    practice is to always call hipSetDevice at the start of HIP coding sequency to establish a known
    standard device.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorDeviceAlreadyInUse
    @see hipGetDevice, hipGetDeviceCount
    """
    pass

def hipGetDevice(deviceId) nogil:
    """@brief Return the default device id for the calling host thread.
    @param [out] device *device is written with the default device
    HIP maintains an default device for each thread using thread-local-storage.
    This device is used implicitly for HIP runtime APIs called by this thread.
    hipGetDevice returns in * @p device the default device for the calling host thread.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see hipSetDevice, hipGetDevicesizeBytes
    """
    pass

def hipGetDeviceCount(count) nogil:
    """@brief Return number of compute-capable devices.
    @param [output] count Returns number of compute-capable devices.
    @returns #hipSuccess, #hipErrorNoDevice
    Returns in @p *count the number of devices that have ability to run compute commands.  If there
    are no such devices, then @ref hipGetDeviceCount will return #hipErrorNoDevice. If 1 or more
    devices can be found, then hipGetDeviceCount returns #hipSuccess.
    """
    pass

def hipDeviceGetAttribute(pi,attr,deviceId) nogil:
    """@brief Query for a specific device attribute.
    @param [out] pi pointer to value to return
    @param [in] attr attribute to query
    @param [in] deviceId which device to query for information
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    """
    pass

def hipDeviceGetDefaultMemPool(mem_pool,device) nogil:
    """@brief Returns the default memory pool of the specified device
    @param [out] mem_pool Default memory pool to return
    @param [in] device    Device index for query the default memory pool
    @returns #chipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotSupported
    @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceSetMemPool(device,mem_pool) nogil:
    """@brief Sets the current memory pool of a device
    The memory pool must be local to the specified device.
    @p hipMallocAsync allocates from the current mempool of the provided stream's device.
    By default, a device's current memory pool is its default memory pool.
    @note Use @p hipMallocFromPoolAsync for asynchronous memory allocations from a device
    different than the one the stream runs on.
    @param [in] device   Device index for the update
    @param [in] mem_pool Memory pool for update as the current on the specified device
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice, #hipErrorNotSupported
    @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceGetMemPool(mem_pool,device) nogil:
    """@brief Gets the current memory pool for the specified device
    Returns the last pool provided to @p hipDeviceSetMemPool for this device
    or the device's default memory pool if @p hipDeviceSetMemPool has never been called.
    By default the current mempool is the default mempool for a device,
    otherwise the returned pool must have been set with @p hipDeviceSetMemPool.
    @param [out] mem_pool Current memory pool on the specified device
    @param [in] device    Device index to query the current memory pool
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGetDeviceProperties(prop,deviceId) nogil:
    """@brief Returns device properties.
    @param [out] prop written with device properties
    @param [in]  deviceId which device to query for information
    @return #hipSuccess, #hipErrorInvalidDevice
    @bug HCC always returns 0 for maxThreadsPerMultiProcessor
    @bug HCC always returns 0 for regsPerBlock
    @bug HCC always returns 0 for l2CacheSize
    Populates hipGetDeviceProperties with information for the specified device.
    """
    pass

def hipDeviceSetCacheConfig(cacheConfig) nogil:
    """@brief Set L1/Shared cache partition.
    @param [in] cacheConfig
    @returns #hipSuccess, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    on those architectures.
    """
    pass

def hipDeviceGetCacheConfig(cacheConfig) nogil:
    """@brief Get Cache configuration for a specific Device
    @param [out] cacheConfig
    @returns #hipSuccess, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    on those architectures.
    """
    pass

def hipDeviceGetLimit(pValue,limit) nogil:
    """@brief Get Resource limits of current device
    @param [out] pValue
    @param [in]  limit
    @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
    Note: Currently, only hipLimitMallocHeapSize is available
    """
    pass

def hipDeviceSetLimit(limit,value) nogil:
    """@brief Set Resource limits of current device
    @param [in] limit
    @param [in] value
    @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
    """
    pass

def hipDeviceGetSharedMemConfig(pConfig) nogil:
    """@brief Returns bank width of shared memory for current device
    @param [out] pConfig
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipGetDeviceFlags(flags) nogil:
    """@brief Gets the flags set for current device
    @param [out] flags
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    """
    pass

def hipDeviceSetSharedMemConfig(config) nogil:
    """@brief The bank width of shared memory on current device is set
    @param [in] config
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipSetDeviceFlags(flags) nogil:
    """@brief The current device behavior is changed according the flags passed.
    @param [in] flags
    The schedule flags impact how HIP waits for the completion of a command running on a device.
    hipDeviceScheduleSpin         : HIP runtime will actively spin in the thread which submitted the
    work until the command completes.  This offers the lowest latency, but will consume a CPU core
    and may increase power. hipDeviceScheduleYield        : The HIP runtime will yield the CPU to
    system so that other tasks can use it.  This may increase latency to detect the completion but
    will consume less power and is friendlier to other tasks in the system.
    hipDeviceScheduleBlockingSync : On ROCm platform, this is a synonym for hipDeviceScheduleYield.
    hipDeviceScheduleAuto         : Use a hueristic to select between Spin and Yield modes.  If the
    number of HIP contexts is greater than the number of logical processors in the system, use Spin
    scheduling.  Else use Yield scheduling.
    hipDeviceMapHost              : Allow mapping host memory.  On ROCM, this is always allowed and
    the flag is ignored. hipDeviceLmemResizeToMax      : @warning ROCm silently ignores this flag.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorSetOnActiveProcess
    """
    pass

def hipChooseDevice(device,prop) nogil:
    """@brief Device which matches hipDeviceProp_t is returned
    @param [out] device ID
    @param [in]  device properties pointer
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipExtGetLinkTypeAndHopCount(device1,device2,linktype,hopcount) nogil:
    """@brief Returns the link type and hop count between two devices
    @param [in] device1 Ordinal for device1
    @param [in] device2 Ordinal for device2
    @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
    @param [out] hopcount Returns the hop count between the two devices
    Queries and returns the HSA link type and the hop count between the two specified devices.
    @returns #hipSuccess, #hipInvalidDevice, #hipErrorRuntimeOther
    """
    pass

def hipIpcGetMemHandle(handle,devPtr) nogil:
    """@brief Gets an interprocess memory handle for an existing device memory
    allocation
    Takes a pointer to the base of an existing device memory allocation created
    with hipMalloc and exports it for use in another process. This is a
    lightweight operation and may be called multiple times on an allocation
    without adverse effects.
    If a region of memory is freed with hipFree and a subsequent call
    to hipMalloc returns memory with the same device address,
    hipIpcGetMemHandle will return a unique handle for the
    new memory.
    @param handle - Pointer to user allocated hipIpcMemHandle to return
    the handle in.
    @param devPtr - Base pointer to previously allocated device memory
    @returns
    hipSuccess,
    hipErrorInvalidHandle,
    hipErrorOutOfMemory,
    hipErrorMapFailed,
    """
    pass

def hipIpcOpenMemHandle(devPtr,handle,flags) nogil:
    """@brief Opens an interprocess memory handle exported from another process
    and returns a device pointer usable in the local process.
    Maps memory exported from another process with hipIpcGetMemHandle into
    the current device address space. For contexts on different devices
    hipIpcOpenMemHandle can attempt to enable peer access between the
    devices as if the user called hipDeviceEnablePeerAccess. This behavior is
    controlled by the hipIpcMemLazyEnablePeerAccess flag.
    hipDeviceCanAccessPeer can determine if a mapping is possible.
    Contexts that may open hipIpcMemHandles are restricted in the following way.
    hipIpcMemHandles from each device in a given process may only be opened
    by one context per device per other process.
    Memory returned from hipIpcOpenMemHandle must be freed with
    hipIpcCloseMemHandle.
    Calling hipFree on an exported memory region before calling
    hipIpcCloseMemHandle in the importing context will result in undefined
    behavior.
    @param devPtr - Returned device pointer
    @param handle - hipIpcMemHandle to open
    @param flags  - Flags for this operation. Must be specified as hipIpcMemLazyEnablePeerAccess
    @returns
    hipSuccess,
    hipErrorMapFailed,
    hipErrorInvalidHandle,
    hipErrorTooManyPeers
    @note During multiple processes, using the same memory handle opened by the current context,
    there is no guarantee that the same device poiter will be returned in @p *devPtr.
    This is diffrent from CUDA.
    """
    pass

def hipIpcCloseMemHandle(devPtr) nogil:
    """@brief Close memory mapped with hipIpcOpenMemHandle
    Unmaps memory returnd by hipIpcOpenMemHandle. The original allocation
    in the exporting process as well as imported mappings in other processes
    will be unaffected.
    Any resources used to enable peer access will be freed if this is the
    last mapping using them.
    @param devPtr - Device pointer returned by hipIpcOpenMemHandle
    @returns
    hipSuccess,
    hipErrorMapFailed,
    hipErrorInvalidHandle,
    """
    pass

def hipIpcGetEventHandle(handle,event) nogil:
    """@brief Gets an opaque interprocess handle for an event.
    This opaque handle may be copied into other processes and opened with hipIpcOpenEventHandle.
    Then hipEventRecord, hipEventSynchronize, hipStreamWaitEvent and hipEventQuery may be used in
    either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
    will result in undefined behavior.
    @param[out]  handle Pointer to hipIpcEventHandle to return the opaque event handle
    @param[in]   event  Event allocated with hipEventInterprocess and hipEventDisableTiming flags
    @returns #hipSuccess, #hipErrorInvalidConfiguration, #hipErrorInvalidValue
    """
    pass

def hipIpcOpenEventHandle(event,handle) nogil:
    """@brief Opens an interprocess event handles.
    Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle. The returned
    hipEvent_t behaves like a locally created event with the hipEventDisableTiming flag specified. This event
    need be freed with hipEventDestroy. Operations on the imported event after the exported event has been freed
    with hipEventDestroy will result in undefined behavior. If the function is called within the same process where
    handle is returned by hipIpcGetEventHandle, it will return hipErrorInvalidContext.
    @param[out]  event  Pointer to hipEvent_t to return the event
    @param[in]   handle The opaque interprocess handle to open
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext
    """
    pass

def hipFuncSetAttribute(func,attr,value) nogil:
    """@}
    @defgroup Execution Execution Control
    @{
    This section describes the execution control functions of HIP runtime API.
    @brief Set attribute for a specific function
    @param [in] func;
    @param [in] attr;
    @param [in] value;
    @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipFuncSetCacheConfig(func,config) nogil:
    """@brief Set Cache configuration for a specific function
    @param [in] config;
    @returns #hipSuccess, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    on those architectures.
    """
    pass

def hipFuncSetSharedMemConfig(func,config) nogil:
    """@brief Set shared memory configuation for a specific function
    @param [in] func
    @param [in] config
    @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipGetLastError() nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Error Error Handling
    @{
    This section describes the error handling functions of HIP runtime API.
    @brief Return last error returned by any HIP runtime API call and resets the stored error code to
    #hipSuccess
    @returns return code from last HIP called from the active host thread
    Returns the last error that has been returned by any of the runtime calls in the same host
    thread, and then resets the saved error to #hipSuccess.
    @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipPeekAtLastError() nogil:
    """@brief Return last error returned by any HIP runtime API call.
    @return #hipSuccess
    Returns the last error that has been returned by any of the runtime calls in the same host
    thread. Unlike hipGetLastError, this function does not reset the saved error code.
    @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipGetErrorName(hip_error) nogil:
    """@brief Return hip error as text string form.
    @param hip_error Error code to convert to name.
    @return const char pointer to the NULL-terminated error name
    @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipGetErrorString(hipError) nogil:
    """@brief Return handy text string message to explain the error which occurred
    @param hipError Error code to convert to string.
    @return const char pointer to the NULL-terminated error string
    @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipDrvGetErrorName(hipError,errorString) nogil:
    """@brief Return hip error as text string form.
    @param [in] hipError Error code to convert to string.
    @param [out] const char pointer to the NULL-terminated error string
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipDrvGetErrorString(hipError,errorString) nogil:
    """@brief Return handy text string message to explain the error which occurred
    @param [in] hipError Error code to convert to string.
    @param [out] const char pointer to the NULL-terminated error string
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipStreamCreate(stream) nogil:
    """@brief Create an asynchronous stream.
    @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
    newly created stream.
    @return #hipSuccess, #hipErrorInvalidValue
    Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
    reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
    the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
    used by the stream, applicaiton must call hipStreamDestroy.
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipStreamCreateWithFlags(stream,flags) nogil:
    """@brief Create an asynchronous stream.
    @param[in, out] stream Pointer to new stream
    @param[in ] flags to control stream creation.
    @return #hipSuccess, #hipErrorInvalidValue
    Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
    reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
    the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
    used by the stream, applicaiton must call hipStreamDestroy. Flags controls behavior of the
    stream.  See #hipStreamDefault, #hipStreamNonBlocking.
    @see hipStreamCreate, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipStreamCreateWithPriority(stream,flags,priority) nogil:
    """@brief Create an asynchronous stream with the specified priority.
    @param[in, out] stream Pointer to new stream
    @param[in ] flags to control stream creation.
    @param[in ] priority of the stream. Lower numbers represent higher priorities.
    @return #hipSuccess, #hipErrorInvalidValue
    Create a new asynchronous stream with the specified priority.  @p stream returns an opaque handle
    that can be used to reference the newly created stream in subsequent hipStream* commands.  The
    stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
    To release the memory used by the stream, applicaiton must call hipStreamDestroy. Flags controls
    behavior of the stream.  See #hipStreamDefault, #hipStreamNonBlocking.
    @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipDeviceGetStreamPriorityRange(leastPriority,greatestPriority) nogil:
    """@brief Returns numerical values that correspond to the least and greatest stream priority.
    @param[in, out] leastPriority pointer in which value corresponding to least priority is returned.
    @param[in, out] greatestPriority pointer in which value corresponding to greatest priority is returned.
    Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least
    and greatest stream priority respectively. Stream priorities follow a convention where lower numbers
    imply greater priorities. The range of meaningful stream priorities is given by
    [*greatestPriority, *leastPriority]. If the user attempts to create a stream with a priority value
    that is outside the the meaningful range as specified by this API, the priority is automatically
    clamped to within the valid range.
    """
    pass

def hipStreamDestroy(stream) nogil:
    """@brief Destroys the specified stream.
    @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
    newly created stream.
    @return #hipSuccess #hipErrorInvalidHandle
    Destroys the specified stream.
    If commands are still executing on the specified stream, some may complete execution before the
    queue is deleted.
    The queue may be destroyed while some commands are still inflight, or may wait for all commands
    queued to the stream before destroying it.
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamQuery, hipStreamWaitEvent,
    hipStreamSynchronize
    """
    pass

def hipStreamQuery(stream) nogil:
    """@brief Return #hipSuccess if all of the operations in the specified @p stream have completed, or
    #hipErrorNotReady if not.
    @param[in] stream stream to query
    @return #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle
    This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
    host threads are sending work to the stream, the status may change immediately after the function
    is called.  It is typically used for debug.
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamSynchronize,
    hipStreamDestroy
    """
    pass

def hipStreamSynchronize(stream) nogil:
    """@brief Wait for all commands in stream to complete.
    @param[in] stream stream identifier.
    @return #hipSuccess, #hipErrorInvalidHandle
    This command is host-synchronous : the host will block until the specified stream is empty.
    This command follows standard null-stream semantics.  Specifically, specifying the null stream
    will cause the command to wait for other streams on the same device to complete all pending
    operations.
    This command honors the hipDeviceLaunchBlocking flag, which controls whether the wait is active
    or blocking.
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipStreamWaitEvent(stream,event,flags) nogil:
    """@brief Make the specified compute stream wait for an event
    @param[in] stream stream to make wait.
    @param[in] event event to wait on
    @param[in] flags control operation [must be 0]
    @return #hipSuccess, #hipErrorInvalidHandle
    This function inserts a wait operation into the specified stream.
    All future work submitted to @p stream will wait until @p event reports completion before
    beginning execution.
    This function only waits for commands in the current stream to complete.  Notably,, this function
    does not impliciy wait for commands in the default stream to complete, even if the specified
    stream is created with hipStreamNonBlocking = 0.
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamDestroy
    """
    pass

def hipStreamGetFlags(stream,flags) nogil:
    """@brief Return flags associated with this stream.
    @param[in] stream stream to be queried
    @param[in,out] flags Pointer to an unsigned integer in which the stream's flags are returned
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
    @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
    Return flags associated with this stream in *@p flags.
    @see hipStreamCreateWithFlags
    """
    pass

def hipStreamGetPriority(stream,priority) nogil:
    """@brief Query the priority of a stream.
    @param[in] stream stream to be queried
    @param[in,out] priority Pointer to an unsigned integer in which the stream's priority is returned
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
    @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
    Query the priority of a stream. The priority is returned in in priority.
    @see hipStreamCreateWithFlags
    """
    pass

def hipExtStreamCreateWithCUMask(stream,cuMaskSize,cuMask) nogil:
    """@brief Create an asynchronous stream with the specified CU mask.
    @param[in, out] stream Pointer to new stream
    @param[in ] cuMaskSize Size of CU mask bit array passed in.
    @param[in ] cuMask Bit-vector representing the CU mask. Each active bit represents using one CU.
    The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
    CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are ignored.
    It is user's responsibility to make sure the input is meaningful.
    @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
    Create a new asynchronous stream with the specified CU mask.  @p stream returns an opaque handle
    that can be used to reference the newly created stream in subsequent hipStream* commands.  The
    stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
    To release the memory used by the stream, application must call hipStreamDestroy.
    @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipExtStreamGetCUMask(stream,cuMaskSize,cuMask) nogil:
    """@brief Get CU mask associated with an asynchronous stream
    @param[in] stream stream to be queried
    @param[in] cuMaskSize number of the block of memories (uint32_t *) allocated by user
    @param[out] cuMask Pointer to a pre-allocated block of memories (uint32_t *) in which
    the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
    each active bit represents one active CU
    @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
    @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    pass

def hipStreamAddCallback(stream,callback,userData,flags) nogil:
    """@brief Adds a callback to be called on the host after all currently enqueued
    items in the stream have completed.  For each
    hipStreamAddCallback call, a callback will be executed exactly once.
    The callback will block later work in the stream until it is finished.
    @param[in] stream   - Stream to add callback to
    @param[in] callback - The function to call once preceding stream operations are complete
    @param[in] userData - User specified data to be passed to the callback function
    @param[in] flags    - Reserved for future use, must be 0
    @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorNotSupported
    @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamSynchronize,
    hipStreamWaitEvent, hipStreamDestroy, hipStreamCreateWithPriority
    """
    pass

def hipStreamWaitValue32(stream,ptr,value,flags,mask) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup StreamM Stream Memory Operations
    @{
    This section describes Stream Memory Wait and Write functions of HIP runtime API.
    @brief Enqueues a wait command to the stream.[BETA]
    @param [in] stream - Stream identifier
    @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
    @param [in] value  - Value to be used in compare operation
    @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
    hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor
    @param [in] mask   - Mask to be applied on value at memory before it is compared with value,
    default value is set to enable every bit
    @returns #hipSuccess, #hipErrorInvalidValue
    Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
    not execute until the defined wait condition is true.
    hipStreamWaitValueGte: waits until *ptr&mask >= value
    hipStreamWaitValueEq : waits until *ptr&mask == value
    hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
    hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
    @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
    @note Support for hipStreamWaitValue32 can be queried using 'hipDeviceGetAttribute()' and
    'hipDeviceAttributeCanUseStreamWaitValue' flag.
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue64, hipStreamWriteValue64,
    hipStreamWriteValue32, hipDeviceGetAttribute
    """
    pass

def hipStreamWaitValue64(stream,ptr,value,flags,mask) nogil:
    """@brief Enqueues a wait command to the stream.[BETA]
    @param [in] stream - Stream identifier
    @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
    @param [in] value  - Value to be used in compare operation
    @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
    hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor.
    @param [in] mask   - Mask to be applied on value at memory before it is compared with value
    default value is set to enable every bit
    @returns #hipSuccess, #hipErrorInvalidValue
    Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
    not execute until the defined wait condition is true.
    hipStreamWaitValueGte: waits until *ptr&mask >= value
    hipStreamWaitValueEq : waits until *ptr&mask == value
    hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
    hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
    @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
    @note Support for hipStreamWaitValue64 can be queried using 'hipDeviceGetAttribute()' and
    'hipDeviceAttributeCanUseStreamWaitValue' flag.
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue32, hipStreamWriteValue64,
    hipStreamWriteValue32, hipDeviceGetAttribute
    """
    pass

def hipStreamWriteValue32(stream,ptr,value,flags) nogil:
    """@brief Enqueues a write command to the stream.[BETA]
    @param [in] stream - Stream identifier
    @param [in] ptr    - Pointer to a GPU accessible memory object
    @param [in] value  - Value to be written
    @param [in] flags  - reserved, ignored for now, will be used in future releases
    @returns #hipSuccess, #hipErrorInvalidValue
    Enqueues a write command to the stream, write operation is performed after all earlier commands
    on this stream have completed the execution.
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
    hipStreamWaitValue64
    """
    pass

def hipStreamWriteValue64(stream,ptr,value,flags) nogil:
    """@brief Enqueues a write command to the stream.[BETA]
    @param [in] stream - Stream identifier
    @param [in] ptr    - Pointer to a GPU accessible memory object
    @param [in] value  - Value to be written
    @param [in] flags  - reserved, ignored for now, will be used in future releases
    @returns #hipSuccess, #hipErrorInvalidValue
    Enqueues a write command to the stream, write operation is performed after all earlier commands
    on this stream have completed the execution.
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
    hipStreamWaitValue64
    """
    pass

def hipEventCreateWithFlags(event,flags) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Event Event Management
    @{
    This section describes the event management functions of HIP runtime API.
    @brief Create an event with the specified flags
    @param[in,out] event Returns the newly created event.
    @param[in] flags     Flags to control event behavior.  Valid values are #hipEventDefault,
     #hipEventBlockingSync, #hipEventDisableTiming, #hipEventInterprocess
    #hipEventDefault : Default flag.  The event will use active synchronization and will support
     timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a
     CPU to poll on the event.
    #hipEventBlockingSync : The event will use blocking synchronization : if hipEventSynchronize is
     called on this event, the thread will block until the event completes.  This can increase latency
     for the synchroniation but can result in lower power and more resources for other CPU threads.
    #hipEventDisableTiming : Disable recording of timing information. Events created with this flag
     would not record profiling data and provide best performance if used for synchronization.
    #hipEventInterprocess : The event can be used as an interprocess event. hipEventDisableTiming
     flag also must be set when hipEventInterprocess flag is set.
    @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
     #hipErrorLaunchFailure, #hipErrorOutOfMemory
    @see hipEventCreate, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
    """
    pass

def hipEventCreate(event) nogil:
    """Create an event
    @param[in,out] event Returns the newly created event.
    @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
    #hipErrorLaunchFailure, #hipErrorOutOfMemory
    @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize,
    hipEventDestroy, hipEventElapsedTime
    """
    pass

def hipEventRecord(event,stream) nogil:
    """
    """
    pass

def hipEventDestroy(event) nogil:
    """@brief Destroy the specified event.
    @param[in] event Event to destroy.
    @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
    #hipErrorLaunchFailure
    Releases memory associated with the event.  If the event is recording but has not completed
    recording when hipEventDestroy() is called, the function will return immediately and the
    completion_future resources will be released later, when the hipDevice is synchronized.
    @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize, hipEventRecord,
    hipEventElapsedTime
    @returns #hipSuccess
    """
    pass

def hipEventSynchronize(event) nogil:
    """@brief Wait for an event to complete.
    This function will block until the event is ready, waiting for all previous work in the stream
    specified when event was recorded with hipEventRecord().
    If hipEventRecord() has not been called on @p event, this function returns immediately.
    TODO-hip- This function needs to support hipEventBlockingSync parameter.
    @param[in] event Event on which to wait.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized,
    #hipErrorInvalidHandle, #hipErrorLaunchFailure
    @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
    hipEventElapsedTime
    """
    pass

def hipEventElapsedTime(ms,start,stop) nogil:
    """@brief Return the elapsed time between two events.
    @param[out] ms : Return time between start and stop in ms.
    @param[in]   start : Start event.
    @param[in]   stop  : Stop event.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotReady, #hipErrorInvalidHandle,
    #hipErrorNotInitialized, #hipErrorLaunchFailure
    Computes the elapsed time between two events. Time is computed in ms, with
    a resolution of approximately 1 us.
    Events which are recorded in a NULL stream will block until all commands
    on all other streams complete execution, and then record the timestamp.
    Events which are recorded in a non-NULL stream will record their timestamp
    when they reach the head of the specified stream, after all previous
    commands in that stream have completed executing.  Thus the time that
    the event recorded may be significantly after the host calls hipEventRecord().
    If hipEventRecord() has not been called on either event, then #hipErrorInvalidHandle is
    returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
    recorded on one or both events (that is, hipEventQuery() would return #hipErrorNotReady on at
    least one of the events), then #hipErrorNotReady is returned.
    @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
    hipEventSynchronize
    """
    pass

def hipEventQuery(event) nogil:
    """@brief Query event status
    @param[in] event Event to query.
    @returns #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle, #hipErrorInvalidValue,
    #hipErrorNotInitialized, #hipErrorLaunchFailure
    Query the status of the specified event.  This function will return #hipSuccess if all
    commands in the appropriate stream (specified to hipEventRecord()) have completed.  If that work
    has not completed, or if hipEventRecord() was not called on the event, then #hipErrorNotReady is
    returned.
    @see hipEventCreate, hipEventCreateWithFlags, hipEventRecord, hipEventDestroy,
    hipEventSynchronize, hipEventElapsedTime
    """
    pass

def hipPointerGetAttributes(attributes,ptr) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Memory Memory Management
    @{
    This section describes the memory management functions of HIP runtime API.
    The following CUDA APIs are not currently supported:
    - cudaMalloc3D
    - cudaMalloc3DArray
    - TODO - more 2D, 3D, array APIs here.
    @brief Return attributes for the specified pointer
    @param [out]  attributes  attributes for the specified pointer
    @param [in]   ptr         pointer to get attributes for
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see hipPointerGetAttribute
    """
    pass

def hipPointerGetAttribute(data,attribute,ptr) nogil:
    """@brief Returns information about the specified pointer.[BETA]
    @param [in, out] data     returned pointer attribute value
    @param [in]      atribute attribute to query for
    @param [in]      ptr      pointer to get attributes for
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipPointerGetAttributes
    """
    pass

def hipDrvPointerGetAttributes(numAttributes,attributes,data,ptr) nogil:
    """@brief Returns information about the specified pointer.[BETA]
    @param [in]  numAttributes   number of attributes to query for
    @param [in]  attributes      attributes to query for
    @param [in, out] data        a two-dimensional containing pointers to memory locations
    where the result of each attribute query will be written to
    @param [in]  ptr             pointer to get attributes for
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @see hipPointerGetAttribute
    """
    pass

def hipImportExternalSemaphore(extSem_out,semHandleDesc) nogil:
    """@brief Imports an external semaphore.
    @param[out] extSem_out  External semaphores to be waited on
    @param[in] semHandleDesc Semaphore import handle descriptor
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipSignalExternalSemaphoresAsync(extSemArray,paramsArray,numExtSems,stream) nogil:
    """@brief Signals a set of external semaphore objects.
    @param[in] extSem_out  External semaphores to be waited on
    @param[in] paramsArray Array of semaphore parameters
    @param[in] numExtSems Number of semaphores to wait on
    @param[in] stream Stream to enqueue the wait operations in
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipWaitExternalSemaphoresAsync(extSemArray,paramsArray,numExtSems,stream) nogil:
    """@brief Waits on a set of external semaphore objects
    @param[in] extSem_out  External semaphores to be waited on
    @param[in] paramsArray Array of semaphore parameters
    @param[in] numExtSems Number of semaphores to wait on
    @param[in] stream Stream to enqueue the wait operations in
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipDestroyExternalSemaphore(extSem) nogil:
    """@brief Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.
    @param[in] extSem handle to an external memory object
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipImportExternalMemory(extMem_out,memHandleDesc) nogil:
    """@brief Imports an external memory object.
    @param[out] extMem_out  Returned handle to an external memory object
    @param[in]  memHandleDesc Memory import handle descriptor
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipExternalMemoryGetMappedBuffer(devPtr,extMem,bufferDesc) nogil:
    """@brief Maps a buffer onto an imported memory object.
    @param[out] devPtr Returned device pointer to buffer
    @param[in]  extMem  Handle to external memory object
    @param[in]  bufferDesc  Buffer descriptor
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipDestroyExternalMemory(extMem) nogil:
    """@brief Destroys an external memory object.
    @param[in] extMem  External memory object to be destroyed
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipMalloc(ptr,size) nogil:
    """@brief Allocate memory on the default accelerator
    @param[out] ptr Pointer to the allocated memory
    @param[in]  size Requested memory size
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
    @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
    hipHostFree, hipHostMalloc
    """
    pass

def hipExtMallocWithFlags(ptr,sizeBytes,flags) nogil:
    """@brief Allocate memory on the default accelerator
    @param[out] ptr Pointer to the allocated memory
    @param[in]  size Requested memory size
    @param[in]  flags Type of memory allocation
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
    @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
    hipHostFree, hipHostMalloc
    """
    pass

def hipMallocHost(ptr,size) nogil:
    """@brief Allocate pinned host memory [Deprecated]
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @deprecated use hipHostMalloc() instead
    """
    pass

def hipMemAllocHost(ptr,size) nogil:
    """@brief Allocate pinned host memory [Deprecated]
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @deprecated use hipHostMalloc() instead
    """
    pass

def hipHostMalloc(ptr,size,flags) nogil:
    """@brief Allocate device accessible page locked host memory
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    @param[in]  flags Type of host memory allocation
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @see hipSetDeviceFlags, hipHostFree
    """
    pass

def hipMallocManaged(dev_ptr,size,flags) nogil:
    """-------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @addtogroup MemoryM Managed Memory
    @{
    @ingroup Memory
    This section describes the managed memory management functions of HIP runtime API.
    @brief Allocates memory that will be automatically managed by HIP.
    @param [out] dev_ptr - pointer to allocated device memory
    @param [in]  size    - requested allocation size in bytes
    @param [in]  flags   - must be either hipMemAttachGlobal or hipMemAttachHost
    (defaults to hipMemAttachGlobal)
    @returns #hipSuccess, #hipErrorMemoryAllocation, #hipErrorNotSupported, #hipErrorInvalidValue
    """
    pass

def hipMemPrefetchAsync(dev_ptr,count,device,stream) nogil:
    """@brief Prefetches memory to the specified destination device using HIP.
    @param [in] dev_ptr  pointer to be prefetched
    @param [in] count    size in bytes for prefetching
    @param [in] device   destination device to prefetch to
    @param [in] stream   stream to enqueue prefetch operation
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemAdvise(dev_ptr,count,advice,device) nogil:
    """@brief Advise about the usage of a given memory range to HIP.
    @param [in] dev_ptr  pointer to memory to set the advice for
    @param [in] count    size in bytes of the memory range
    @param [in] advice   advice to be applied for the specified memory range
    @param [in] device   device to apply the advice for
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemRangeGetAttribute(data,data_size,attribute,dev_ptr,count) nogil:
    """@brief Query an attribute of a given memory range in HIP.
    @param [in,out] data   a pointer to a memory location where the result of each
    attribute query will be written to
    @param [in] data_size  the size of data
    @param [in] attribute  the attribute to query
    @param [in] dev_ptr    start of the range to query
    @param [in] count      size of the range to query
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemRangeGetAttributes(data,data_sizes,attributes,num_attributes,dev_ptr,count) nogil:
    """@brief Query attributes of a given memory range in HIP.
    @param [in,out] data     a two-dimensional array containing pointers to memory locations
    where the result of each attribute query will be written to
    @param [in] data_sizes   an array, containing the sizes of each result
    @param [in] attributes   the attribute to query
    @param [in] num_attributes  an array of attributes to query (numAttributes and the number
    of attributes in this array should match)
    @param [in] dev_ptr      start of the range to query
    @param [in] count        size of the range to query
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipStreamAttachMemAsync(stream,dev_ptr,length,flags) nogil:
    """@brief Attach memory to a stream asynchronously in HIP.
    @param [in] stream     - stream in which to enqueue the attach operation
    @param [in] dev_ptr    - pointer to memory (must be a pointer to managed memory or
    to a valid host-accessible region of system-allocated memory)
    @param [in] length     - length of memory (defaults to zero)
    @param [in] flags      - must be one of hipMemAttachGlobal, hipMemAttachHost or
    hipMemAttachSingle (defaults to hipMemAttachSingle)
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMallocAsync(dev_ptr,size,stream) nogil:
    """@brief Allocates memory with stream ordered semantics
    Inserts a memory allocation operation into @p stream.
    A pointer to the allocated memory is returned immediately in *dptr.
    The allocation must not be accessed until the the allocation operation completes.
    The allocation comes from the memory pool associated with the stream's device.
    @note The default memory pool of a device contains device memory from that device.
    @note Basic stream ordering allows future work submitted into the same stream to use the allocation.
    Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
    operation completes before work submitted in a separate stream runs.
    @note During stream capture, this function results in the creation of an allocation node. In this case,
    the allocation is owned by the graph instead of the memory pool. The memory pool's properties
    are used to set the node's creation parameters.
    @param [out] dev_ptr  Returned device pointer of memory allocation
    @param [in] size      Number of bytes to allocate
    @param [in] stream    The stream establishing the stream ordering contract and
    the memory pool to allocate from
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
    @see hipMallocFromPoolAsync, hipFreeAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipFreeAsync(dev_ptr,stream) nogil:
    """@brief Frees memory with stream ordered semantics
    Inserts a free operation into @p stream.
    The allocation must not be used after stream execution reaches the free.
    After this API returns, accessing the memory from any subsequent work launched on the GPU
    or querying its pointer attributes results in undefined behavior.
    @note During stream capture, this function results in the creation of a free node and
    must therefore be passed the address of a graph allocation.
    @param [in] dev_ptr Pointer to device memory to free
    @param [in] stream  The stream, where the destruciton will occur according to the execution order
    @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported
    @see hipMallocFromPoolAsync, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolTrimTo(mem_pool,min_bytes_to_hold) nogil:
    """@brief Releases freed memory back to the OS
    Releases memory back to the OS until the pool contains fewer than @p min_bytes_to_keep
    reserved bytes, or there is no more memory that the allocator can safely release.
    The allocator cannot release OS allocations that back outstanding asynchronous allocations.
    The OS allocations may happen at different granularity from the user allocations.
    @note: Allocations that have not been freed count as outstanding.
    @note: Allocations that have been asynchronously freed but whose completion has
    not been observed on the host (eg. by a synchronize) can count as outstanding.
    @param[in] mem_pool          The memory pool to trim allocations
    @param[in] min_bytes_to_hold If the pool has less than min_bytes_to_hold reserved,
    then the TrimTo operation is a no-op.  Otherwise the memory pool will contain
    at least min_bytes_to_hold bytes reserved after the operation.
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolSetAttribute(mem_pool,attr,value) nogil:
    """@brief Sets attributes of a memory pool
    Supported attributes are:
    - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
    Amount of reserved memory in bytes to hold onto before trying
    to release memory back to the OS. When more than the release
    threshold bytes of memory are held by the memory pool, the
    allocator will try to release memory back to the OS on the
    next call to stream, event or context synchronize. (default 0)
    - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
    Allow @p hipMallocAsync to use memory asynchronously freed
    in another stream as long as a stream ordering dependency
    of the allocating stream on the free action exists.
    HIP events and null stream interactions can create the required
    stream ordered dependencies. (default enabled)
    - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
    Allow reuse of already completed frees when there is no dependency
    between the free and allocation. (default enabled)
    - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
    Allow @p hipMallocAsync to insert new stream dependencies
    in order to establish the stream ordering required to reuse
    a piece of memory released by @p hipFreeAsync (default enabled).
    @param [in] mem_pool The memory pool to modify
    @param [in] attr     The attribute to modify
    @param [in] value    Pointer to the value to assign
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolGetAttribute(mem_pool,attr,value) nogil:
    """@brief Gets attributes of a memory pool
    Supported attributes are:
    - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
    Amount of reserved memory in bytes to hold onto before trying
    to release memory back to the OS. When more than the release
    threshold bytes of memory are held by the memory pool, the
    allocator will try to release memory back to the OS on the
    next call to stream, event or context synchronize. (default 0)
    - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
    Allow @p hipMallocAsync to use memory asynchronously freed
    in another stream as long as a stream ordering dependency
    of the allocating stream on the free action exists.
    HIP events and null stream interactions can create the required
    stream ordered dependencies. (default enabled)
    - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
    Allow reuse of already completed frees when there is no dependency
    between the free and allocation. (default enabled)
    - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
    Allow @p hipMallocAsync to insert new stream dependencies
    in order to establish the stream ordering required to reuse
    a piece of memory released by @p hipFreeAsync (default enabled).
    @param [in] mem_pool The memory pool to get attributes of
    @param [in] attr     The attribute to get
    @param [in] value    Retrieved value
    @returns  #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolSetAccess(mem_pool,desc_list,count) nogil:
    """@brief Controls visibility of the specified pool between devices
    @param [in] mem_pool   Memory pool for acccess change
    @param [in] desc_list  Array of access descriptors. Each descriptor instructs the access to enable for a single gpu
    @param [in] count  Number of descriptors in the map array.
    @returns  #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolGetAccess(flags,mem_pool,location) nogil:
    """@brief Returns the accessibility of a pool from a device
    Returns the accessibility of the pool's memory from the specified location.
    @param [out] flags    Accessibility of the memory pool from the specified location/device
    @param [in] mem_pool   Memory pool being queried
    @param [in] location  Location/device for memory pool access
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolCreate(mem_pool,pool_props) nogil:
    """@brief Creates a memory pool
    Creates a HIP memory pool and returns the handle in @p mem_pool. The @p pool_props determines
    the properties of the pool such as the backing device and IPC capabilities.
    By default, the memory pool will be accessible from the device it is allocated on.
    @param [out] mem_pool    Contains createed memory pool
    @param [in] pool_props   Memory pool properties
    @note Specifying hipMemHandleTypeNone creates a memory pool that will not support IPC.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolDestroy,
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolDestroy(mem_pool) nogil:
    """@brief Destroys the specified memory pool
    If any pointers obtained from this pool haven't been freed or
    the pool has free operations that haven't completed
    when @p hipMemPoolDestroy is invoked, the function will return immediately and the
    resources associated with the pool will be released automatically
    once there are no more outstanding allocations.
    Destroying the current mempool of a device sets the default mempool of
    that device as the current mempool for that device.
    @param [in] mem_pool Memory pool for destruction
    @note A device's default memory pool cannot be destroyed.
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMallocFromPoolAsync(dev_ptr,size,mem_pool,stream) nogil:
    """@brief Allocates memory from a specified pool with stream ordered semantics.
    Inserts an allocation operation into @p stream.
    A pointer to the allocated memory is returned immediately in @p dev_ptr.
    The allocation must not be accessed until the the allocation operation completes.
    The allocation comes from the specified memory pool.
    @note The specified memory pool may be from a device different than that of the specified @p stream.
    Basic stream ordering allows future work submitted into the same stream to use the allocation.
    Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
    operation completes before work submitted in a separate stream runs.
    @note During stream capture, this function results in the creation of an allocation node. In this case,
    the allocation is owned by the graph instead of the memory pool. The memory pool's properties
    are used to set the node's creation parameters.
    @param [out] dev_ptr Returned device pointer
    @param [in] size     Number of bytes to allocate
    @param [in] mem_pool The pool to allocate from
    @param [in] stream   The stream establishing the stream ordering semantic
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
    @see hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
    hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess,
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolExportToShareableHandle(shared_handle,mem_pool,handle_type,flags) nogil:
    """@brief Exports a memory pool to the requested handle type.
    Given an IPC capable mempool, create an OS handle to share the pool with another process.
    A recipient process can convert the shareable handle into a mempool with @p hipMemPoolImportFromShareableHandle.
    Individual pointers can then be shared with the @p hipMemPoolExportPointer and @p hipMemPoolImportPointer APIs.
    The implementation of what the shareable handle is and how it can be transferred is defined by the requested
    handle type.
    @note: To create an IPC capable mempool, create a mempool with a @p hipMemAllocationHandleType other
    than @p hipMemHandleTypeNone.
    @param [out] shared_handle Pointer to the location in which to store the requested handle
    @param [in] mem_pool       Pool to export
    @param [in] handle_type    The type of handle to create
    @param [in] flags          Must be 0
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    @see hipMemPoolImportFromShareableHandle
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolImportFromShareableHandle(mem_pool,shared_handle,handle_type,flags) nogil:
    """@brief Imports a memory pool from a shared handle.
    Specific allocations can be imported from the imported pool with @p hipMemPoolImportPointer.
    @note Imported memory pools do not support creating new allocations.
    As such imported memory pools may not be used in @p hipDeviceSetMemPool
    or @p hipMallocFromPoolAsync calls.
    @param [out] mem_pool     Returned memory pool
    @param [in] shared_handle OS handle of the pool to open
    @param [in] handle_type   The type of handle being imported
    @param [in] flags         Must be 0
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    @see hipMemPoolExportToShareableHandle
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolExportPointer(export_data,dev_ptr) nogil:
    """@brief Export data to share a memory pool allocation between processes.
    Constructs @p export_data for sharing a specific allocation from an already shared memory pool.
    The recipient process can import the allocation with the @p hipMemPoolImportPointer api.
    The data is not a handle and may be shared through any IPC mechanism.
    @param[out] export_data  Returned export data
    @param[in] dev_ptr       Pointer to memory being exported
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    @see hipMemPoolImportPointer
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemPoolImportPointer(dev_ptr,mem_pool,export_data) nogil:
    """@brief Import a memory pool allocation from another process.
    Returns in @p dev_ptr a pointer to the imported memory.
    The imported memory must not be accessed before the allocation operation completes
    in the exporting process. The imported memory must be freed from all importing processes before
    being freed in the exporting process. The pointer may be freed with @p hipFree
    or @p hipFreeAsync. If @p hipFreeAsync is used, the free must be completed
    on the importing process before the free operation on the exporting process.
    @note The @p hipFreeAsync api may be used in the exporting process before
    the @p hipFreeAsync operation completes in its stream as long as the
    @p hipFreeAsync in the exporting process specifies a stream with
    a stream dependency on the importing process's @p hipFreeAsync.
    @param [out] dev_ptr     Pointer to imported memory
    @param [in] mem_pool     Memory pool from which to import a pointer
    @param [in] export_data  Data specifying the memory to import
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized, #hipErrorOutOfMemory
    @see hipMemPoolExportPointer
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipHostAlloc(ptr,size,flags) nogil:
    """@brief Allocate device accessible page locked host memory [Deprecated]
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    @param[in]  flags Type of host memory allocation
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @deprecated use hipHostMalloc() instead
    """
    pass

def hipHostGetDevicePointer(devPtr,hstPtr,flags) nogil:
    """@brief Get Device pointer from Host Pointer allocated through hipHostMalloc
    @param[out] dstPtr Device Pointer mapped to passed host pointer
    @param[in]  hstPtr Host Pointer allocated through hipHostMalloc
    @param[in]  flags Flags to be passed for extension
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    @see hipSetDeviceFlags, hipHostMalloc
    """
    pass

def hipHostGetFlags(flagsPtr,hostPtr) nogil:
    """@brief Return flags associated with host pointer
    @param[out] flagsPtr Memory location to store flags
    @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipHostMalloc
    """
    pass

def hipHostRegister(hostPtr,sizeBytes,flags) nogil:
    """@brief Register host memory so it can be accessed from the current device.
    @param[out] hostPtr Pointer to host memory to be registered.
    @param[in] sizeBytes size of the host memory
    @param[in] flags.  See below.
    Flags:
    - #hipHostRegisterDefault   Memory is Mapped and Portable
    - #hipHostRegisterPortable  Memory is considered registered by all contexts.  HIP only supports
    one context so this is always assumed true.
    - #hipHostRegisterMapped    Map the allocation into the address space for the current device.
    The device pointer can be obtained with #hipHostGetDevicePointer.
    After registering the memory, use #hipHostGetDevicePointer to obtain the mapped device pointer.
    On many systems, the mapped device pointer will have a different value than the mapped host
    pointer.  Applications must use the device pointer in device code, and the host pointer in device
    code.
    On some systems, registered memory is pinned.  On some systems, registered memory may not be
    actually be pinned but uses OS or hardware facilities to all GPU access to the host memory.
    Developers are strongly encouraged to register memory blocks which are aligned to the host
    cache-line size. (typically 64-bytes but can be obtains from the CPUID instruction).
    If registering non-aligned pointers, the application must take care when register pointers from
    the same cache line on different devices.  HIP's coarse-grained synchronization model does not
    guarantee correct results if different devices write to different parts of the same cache block -
    typically one of the writes will "win" and overwrite data from the other registered memory
    region.
    @return #hipSuccess, #hipErrorOutOfMemory
    @see hipHostUnregister, hipHostGetFlags, hipHostGetDevicePointer
    """
    pass

def hipHostUnregister(hostPtr) nogil:
    """@brief Un-register host pointer
    @param[in] hostPtr Host pointer previously registered with #hipHostRegister
    @return Error code
    @see hipHostRegister
    """
    pass

def hipMallocPitch(ptr,pitch,width,height) nogil:
    """Allocates at least width (in bytes) * height bytes of linear memory
    Padding may occur to ensure alighnment requirements are met for the given row
    The change in width size due to padding will be returned in *pitch.
    Currently the alignment is set to 128 bytes
    @param[out] ptr Pointer to the allocated device memory
    @param[out] pitch Pitch for allocation (in bytes)
    @param[in]  width Requested pitched allocation width (in bytes)
    @param[in]  height Requested pitched allocation height
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return Error code
    @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
    hipMalloc3DArray, hipHostMalloc
    """
    pass

def hipMemAllocPitch(dptr,pitch,widthInBytes,height,elementSizeBytes) nogil:
    """Allocates at least width (in bytes) * height bytes of linear memory
    Padding may occur to ensure alighnment requirements are met for the given row
    The change in width size due to padding will be returned in *pitch.
    Currently the alignment is set to 128 bytes
    @param[out] dptr Pointer to the allocated device memory
    @param[out] pitch Pitch for allocation (in bytes)
    @param[in]  width Requested pitched allocation width (in bytes)
    @param[in]  height Requested pitched allocation height
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array.
    Given the row and column of an array element of type T, the address is computed as:
    T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
    @return Error code
    @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
    hipMalloc3DArray, hipHostMalloc
    """
    pass

def hipFree(ptr) nogil:
    """@brief Free memory allocated by the hcc hip memory allocation API.
    This API performs an implicit hipDeviceSynchronize() call.
    If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
    @param[in] ptr Pointer to memory to be freed
    @return #hipSuccess
    @return #hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated
    with hipHostMalloc)
    @see hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
    hipMalloc3DArray, hipHostMalloc
    """
    pass

def hipFreeHost(ptr) nogil:
    """@brief Free memory allocated by the hcc hip host memory allocation API.  [Deprecated]
    @param[in] ptr Pointer to memory to be freed
    @return #hipSuccess,
    #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
     hipMalloc)
    @deprecated use hipHostFree() instead
    """
    pass

def hipHostFree(ptr) nogil:
    """@brief Free memory allocated by the hcc hip host memory allocation API
    This API performs an implicit hipDeviceSynchronize() call.
    If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
    @param[in] ptr Pointer to memory to be freed
    @return #hipSuccess,
    #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
    hipMalloc)
    @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D,
    hipMalloc3DArray, hipHostMalloc
    """
    pass

def hipMemcpy(dst,src,sizeBytes,kind) nogil:
    """@brief Copy data from src to dst.
    It supports memory from host to device,
    device to host, device to device and host to host
    The src and dst must not overlap.
    For hipMemcpy, the copy is always performed by the current device (set by hipSetDevice).
    For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the
    device where the src data is physically located. For optimal peer-to-peer copies, the copy device
    must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy
    agent as the current device and src/dest as the peerDevice argument.  if this is not done, the
    hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
    Calling hipMemcpy with dst and src pointers that do not match the hipMemcpyKind results in
    undefined behavior.
    @param[out]  dst Data being copy to
    @param[in]  src Data being copy from
    @param[in]  sizeBytes Data size in bytes
    @param[in]  copyType Memory copy type
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknowni
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyWithStream(dst,src,sizeBytes,kind,stream) nogil:
    """
    """
    pass

def hipMemcpyHtoD(dst,src,sizeBytes) nogil:
    """@brief Copy data from Host to Device
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyDtoH(dst,src,sizeBytes) nogil:
    """@brief Copy data from Device to Host
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyDtoD(dst,src,sizeBytes) nogil:
    """@brief Copy data from Device to Device
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyHtoDAsync(dst,src,sizeBytes,stream) nogil:
    """@brief Copy data from Host to Device asynchronously
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyDtoHAsync(dst,src,sizeBytes,stream) nogil:
    """@brief Copy data from Device to Host asynchronously
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipMemcpyDtoDAsync(dst,src,sizeBytes,stream) nogil:
    """@brief Copy data from Device to Device asynchronously
    @param[out]  dst Data being copy to
    @param[in]   src Data being copy from
    @param[in]   sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
    #hipErrorInvalidValue
    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    hipMemHostAlloc, hipMemHostGetDevicePointer
    """
    pass

def hipModuleGetGlobal(dptr,bytes,hmod,name) nogil:
    """@brief Returns a global pointer from a module.
    Returns in *dptr and *bytes the pointer and size of the global of name name located in module hmod.
    If no variable of that name exists, it returns hipErrorNotFound. Both parameters dptr and bytes are optional.
    If one of them is NULL, it is ignored and hipSuccess is returned.
    @param[out]  dptr  Returns global device pointer
    @param[out]  bytes Returns global size in bytes
    @param[in]   hmod  Module to retrieve global from
    @param[in]   name  Name of global to retrieve
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotFound, #hipErrorInvalidContext
    """
    pass

def hipGetSymbolAddress(devPtr,symbol) nogil:
    """@brief Gets device pointer associated with symbol on the device.
    @param[out]  devPtr  pointer to the device associated the symbole
    @param[in]   symbol  pointer to the symbole of the device
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipGetSymbolSize(size,symbol) nogil:
    """@brief Gets the size of the given symbol on the device.
    @param[in]   symbol  pointer to the device symbole
    @param[out]  size  pointer to the size
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyToSymbol(symbol,src,sizeBytes,offset,kind) nogil:
    """@brief Copies data to the given symbol on the device.
    Symbol HIP APIs allow a kernel to define a device-side data symbol which can be accessed on
    the host side. The symbol can be in __constant or device space.
    Note that the symbol name needs to be encased in the HIP_SYMBOL macro.
    This also applies to hipMemcpyFromSymbol, hipGetSymbolAddress, and hipGetSymbolSize.
    For detail usage, see the example at
    https://github.com/ROCm-Developer-Tools/HIP/blob/rocm-5.0.x/docs/markdown/hip_porting_guide.md
    @param[out]  symbol  pointer to the device symbole
    @param[in]   src  pointer to the source address
    @param[in]   sizeBytes  size in bytes to copy
    @param[in]   offset  offset in bytes from start of symbole
    @param[in]   kind  type of memory transfer
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyToSymbolAsync(symbol,src,sizeBytes,offset,kind,stream) nogil:
    """@brief Copies data to the given symbol on the device asynchronously.
    @param[out]  symbol  pointer to the device symbole
    @param[in]   src  pointer to the source address
    @param[in]   sizeBytes  size in bytes to copy
    @param[in]   offset  offset in bytes from start of symbole
    @param[in]   kind  type of memory transfer
    @param[in]   stream  stream identifier
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyFromSymbol(dst,symbol,sizeBytes,offset,kind) nogil:
    """@brief Copies data from the given symbol on the device.
    @param[out]  dptr  Returns pointer to destinition memory address
    @param[in]   symbol  pointer to the symbole address on the device
    @param[in]   sizeBytes  size in bytes to copy
    @param[in]   offset  offset in bytes from the start of symbole
    @param[in]   kind  type of memory transfer
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyFromSymbolAsync(dst,symbol,sizeBytes,offset,kind,stream) nogil:
    """@brief Copies data from the given symbol on the device asynchronously.
    @param[out]  dptr  Returns pointer to destinition memory address
    @param[in]   symbol  pointer to the symbole address on the device
    @param[in]   sizeBytes  size in bytes to copy
    @param[in]   offset  offset in bytes from the start of symbole
    @param[in]   kind  type of memory transfer
    @param[in]   stream  stream identifier
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyAsync(dst,src,sizeBytes,kind,stream) nogil:
    """@brief Copy data from src to dst asynchronously.
    @warning If host or dest are not pinned, the memory copy will be performed synchronously.  For
    best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.
    @warning on HCC hipMemcpyAsync does not support overlapped H2D and D2H copies.
    For hipMemcpy, the copy is always performed by the device associated with the specified stream.
    For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is a
    attached to the device where the src data is physically located. For optimal peer-to-peer copies,
    the copy device must be able to access the src and dst pointers (by calling
    hipDeviceEnablePeerAccess with copy agent as the current device and src/dest as the peerDevice
    argument.  if this is not done, the hipMemcpy will still work, but will perform the copy using a
    staging buffer on the host.
    @param[out] dst Data being copy to
    @param[in]  src Data being copy from
    @param[in]  sizeBytes Data size in bytes
    @param[in]  accelerator_view Accelerator view which the copy is being enqueued
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
    @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyToSymbol,
    hipMemcpyFromSymbol, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync,
    hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync,
    hipMemcpyFromSymbolAsync
    """
    pass

def hipMemset(dst,value,sizeBytes) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    byte value value.
    @param[out] dst Data being filled
    @param[in]  constant value to be set
    @param[in]  sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD8(dest,value,count) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    byte value value.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD8Async(dest,value,count,stream) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    byte value value.
    hipMemsetD8Async() is asynchronous with respect to the host, so the call may return before the
    memset is complete. The operation can optionally be associated to a stream by passing a non-zero
    stream argument. If stream is non-zero, the operation may overlap with operations in other
    streams.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @param[in]  stream - Stream identifier
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD16(dest,value,count) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    short value value.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD16Async(dest,value,count,stream) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    short value value.
    hipMemsetD16Async() is asynchronous with respect to the host, so the call may return before the
    memset is complete. The operation can optionally be associated to a stream by passing a non-zero
    stream argument. If stream is non-zero, the operation may overlap with operations in other
    streams.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @param[in]  stream - Stream identifier
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD32(dest,value,count) nogil:
    """@brief Fills the memory area pointed to by dest with the constant integer
    value for specified number of times.
    @param[out] dst Data being filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetAsync(dst,value,sizeBytes,stream) nogil:
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant
    byte value value.
    hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the
    memset is complete. The operation can optionally be associated to a stream by passing a non-zero
    stream argument. If stream is non-zero, the operation may overlap with operations in other
    streams.
    @param[out] dst Pointer to device memory
    @param[in]  value - Value to set for each byte of specified memory
    @param[in]  sizeBytes - Size in bytes to set
    @param[in]  stream - Stream identifier
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemsetD32Async(dst,value,count,stream) nogil:
    """@brief Fills the memory area pointed to by dev with the constant integer
    value for specified number of times.
    hipMemsetD32Async() is asynchronous with respect to the host, so the call may return before the
    memset is complete. The operation can optionally be associated to a stream by passing a non-zero
    stream argument. If stream is non-zero, the operation may overlap with operations in other
    streams.
    @param[out] dst Pointer to device memory
    @param[in]  value - Value to set for each byte of specified memory
    @param[in]  count - number of values to be set
    @param[in]  stream - Stream identifier
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset2D(dst,pitch,value,width,height) nogil:
    """@brief Fills the memory area pointed to by dst with the constant value.
    @param[out] dst Pointer to device memory
    @param[in]  pitch - data size in bytes
    @param[in]  value - constant value to be set
    @param[in]  width
    @param[in]  height
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset2DAsync(dst,pitch,value,width,height,stream) nogil:
    """@brief Fills asynchronously the memory area pointed to by dst with the constant value.
    @param[in]  dst Pointer to device memory
    @param[in]  pitch - data size in bytes
    @param[in]  value - constant value to be set
    @param[in]  width
    @param[in]  height
    @param[in]  stream
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset3D(pitchedDevPtr,value,extent) nogil:
    """@brief Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
    @param[in] pitchedDevPtr
    @param[in]  value - constant value to be set
    @param[in]  extent
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset3DAsync(pitchedDevPtr,value,extent,stream) nogil:
    """@brief Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
    @param[in] pitchedDevPtr
    @param[in]  value - constant value to be set
    @param[in]  extent
    @param[in]  stream
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemGetInfo(free,total) nogil:
    """@brief Query memory info.
    Return snapshot of free memory, and total allocatable memory on the device.
    Returns in *free a snapshot of the current free memory.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @warning On HCC, the free memory only accounts for memory allocated by this process and may be
    optimistic.
    """
    pass

def hipMemPtrGetInfo(ptr,size) nogil:
    """
    """
    pass

def hipMallocArray(array,desc,width,height,flags) nogil:
    """@brief Allocate an array on the device.
    @param[out]  array  Pointer to allocated array in device memory
    @param[in]   desc   Requested channel format
    @param[in]   width  Requested array allocation width
    @param[in]   height Requested array allocation height
    @param[in]   flags  Requested properties of allocated array
    @return      #hipSuccess, #hipErrorOutOfMemory
    @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
    """
    pass

def hipArrayCreate(pHandle,pAllocateArray) nogil:
    """
    """
    pass

def hipArrayDestroy(array) nogil:
    """
    """
    pass

def hipArray3DCreate(array,pAllocateArray) nogil:
    """
    """
    pass

def hipMalloc3D(pitchedDevPtr,extent) nogil:
    """
    """
    pass

def hipFreeArray(array) nogil:
    """@brief Frees an array on the device.
    @param[in]  array  Pointer to array to free
    @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
    """
    pass

def hipFreeMipmappedArray(mipmappedArray) nogil:
    """@brief Frees a mipmapped array on the device
    @param[in] mipmappedArray - Pointer to mipmapped array to free
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMalloc3DArray(array,desc,extent,flags) nogil:
    """@brief Allocate an array on the device.
    @param[out]  array  Pointer to allocated array in device memory
    @param[in]   desc   Requested channel format
    @param[in]   extent Requested array allocation width, height and depth
    @param[in]   flags  Requested properties of allocated array
    @return      #hipSuccess, #hipErrorOutOfMemory
    @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
    """
    pass

def hipMallocMipmappedArray(mipmappedArray,desc,extent,numLevels,flags) nogil:
    """@brief Allocate a mipmapped array on the device
    @param[out] mipmappedArray  - Pointer to allocated mipmapped array in device memory
    @param[in]  desc            - Requested channel format
    @param[in]  extent          - Requested allocation size (width field in elements)
    @param[in]  numLevels       - Number of mipmap levels to allocate
    @param[in]  flags           - Flags for extensions
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    """
    pass

def hipGetMipmappedArrayLevel(levelArray,mipmappedArray,level) nogil:
    """@brief Gets a mipmap level of a HIP mipmapped array
    @param[out] levelArray     - Returned mipmap level HIP array
    @param[in]  mipmappedArray - HIP mipmapped array
    @param[in]  level          - Mipmap level
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpy2D(dst,dpitch,src,spitch,width,height,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst    Destination memory address
    @param[in]   dpitch Pitch of destination memory
    @param[in]   src    Source memory address
    @param[in]   spitch Pitch of source memory
    @param[in]   width  Width of matrix transfer (columns in bytes)
    @param[in]   height Height of matrix transfer (rows)
    @param[in]   kind   Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyParam2D(pCopy) nogil:
    """@brief Copies memory for 2D arrays.
    @param[in]   pCopy Parameters for the memory copy
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    hipMemcpyToSymbol, hipMemcpyAsync
    """
    pass

def hipMemcpyParam2DAsync(pCopy,stream) nogil:
    """@brief Copies memory for 2D arrays.
    @param[in]   pCopy Parameters for the memory copy
    @param[in]   stream Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    hipMemcpyToSymbol, hipMemcpyAsync
    """
    pass

def hipMemcpy2DAsync(dst,dpitch,src,spitch,width,height,kind,stream) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst    Destination memory address
    @param[in]   dpitch Pitch of destination memory
    @param[in]   src    Source memory address
    @param[in]   spitch Pitch of source memory
    @param[in]   width  Width of matrix transfer (columns in bytes)
    @param[in]   height Height of matrix transfer (rows)
    @param[in]   kind   Type of transfer
    @param[in]   stream Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy2DToArray(dst,wOffset,hOffset,src,spitch,width,height,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst     Destination memory address
    @param[in]   wOffset Destination starting X offset
    @param[in]   hOffset Destination starting Y offset
    @param[in]   src     Source memory address
    @param[in]   spitch  Pitch of source memory
    @param[in]   width   Width of matrix transfer (columns in bytes)
    @param[in]   height  Height of matrix transfer (rows)
    @param[in]   kind    Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy2DToArrayAsync(dst,wOffset,hOffset,src,spitch,width,height,kind,stream) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst     Destination memory address
    @param[in]   wOffset Destination starting X offset
    @param[in]   hOffset Destination starting Y offset
    @param[in]   src     Source memory address
    @param[in]   spitch  Pitch of source memory
    @param[in]   width   Width of matrix transfer (columns in bytes)
    @param[in]   height  Height of matrix transfer (rows)
    @param[in]   kind    Type of transfer
    @param[in]   stream    Accelerator view which the copy is being enqueued
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyToArray(dst,wOffset,hOffset,src,count,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst     Destination memory address
    @param[in]   wOffset Destination starting X offset
    @param[in]   hOffset Destination starting Y offset
    @param[in]   src     Source memory address
    @param[in]   count   size in bytes to copy
    @param[in]   kind    Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyFromArray(dst,srcArray,wOffset,hOffset,count,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst       Destination memory address
    @param[in]   srcArray  Source memory address
    @param[in]   woffset   Source starting X offset
    @param[in]   hOffset   Source starting Y offset
    @param[in]   count     Size in bytes to copy
    @param[in]   kind      Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy2DFromArray(dst,dpitch,src,wOffset,hOffset,width,height,kind) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst       Destination memory address
    @param[in]   dpitch    Pitch of destination memory
    @param[in]   src       Source memory address
    @param[in]   wOffset   Source starting X offset
    @param[in]   hOffset   Source starting Y offset
    @param[in]   width     Width of matrix transfer (columns in bytes)
    @param[in]   height    Height of matrix transfer (rows)
    @param[in]   kind      Type of transfer
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy2DFromArrayAsync(dst,dpitch,src,wOffset,hOffset,width,height,kind,stream) nogil:
    """@brief Copies data between host and device asynchronously.
    @param[in]   dst       Destination memory address
    @param[in]   dpitch    Pitch of destination memory
    @param[in]   src       Source memory address
    @param[in]   wOffset   Source starting X offset
    @param[in]   hOffset   Source starting Y offset
    @param[in]   width     Width of matrix transfer (columns in bytes)
    @param[in]   height    Height of matrix transfer (rows)
    @param[in]   kind      Type of transfer
    @param[in]   stream    Accelerator view which the copy is being enqueued
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyAtoH(dst,srcArray,srcOffset,count) nogil:
    """@brief Copies data between host and device.
    @param[in]   dst       Destination memory address
    @param[in]   srcArray  Source array
    @param[in]   srcoffset Offset in bytes of source array
    @param[in]   count     Size of memory copy in bytes
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpyHtoA(dstArray,dstOffset,srcHost,count) nogil:
    """@brief Copies data between host and device.
    @param[in]   dstArray   Destination memory address
    @param[in]   dstOffset  Offset in bytes of destination array
    @param[in]   srcHost    Source host pointer
    @param[in]   count      Size of memory copy in bytes
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy3D(p) nogil:
    """@brief Copies data between host and device.
    @param[in]   p   3D memory copy parameters
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy3DAsync(p,stream) nogil:
    """@brief Copies data between host and device asynchronously.
    @param[in]   p        3D memory copy parameters
    @param[in]   stream   Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipDrvMemcpy3D(pCopy) nogil:
    """@brief Copies data between host and device.
    @param[in]   pCopy   3D memory copy parameters
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipDrvMemcpy3DAsync(pCopy,stream) nogil:
    """@brief Copies data between host and device asynchronously.
    @param[in]   pCopy    3D memory copy parameters
    @param[in]   stream   Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipDeviceCanAccessPeer(canAccessPeer,deviceId,peerDeviceId) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup PeerToPeer PeerToPeer Device Memory Access
    @{
    @warning PeerToPeer support is experimental.
    This section describes the PeerToPeer device memory access functions of HIP runtime API.
    @brief Determine if a device can access a peer's memory.
    @param [out] canAccessPeer Returns the peer access capability (0 or 1)
    @param [in] device - device from where memory may be accessed.
    @param [in] peerDevice - device where memory is physically located
    Returns "1" in @p canAccessPeer if the specified @p device is capable
    of directly accessing memory physically located on peerDevice , or "0" if not.
    Returns "0" in @p canAccessPeer if deviceId == peerDeviceId, and both are valid devices : a
    device is not a peer of itself.
    @returns #hipSuccess,
    @returns #hipErrorInvalidDevice if deviceId or peerDeviceId are not valid devices
    """
    pass

def hipDeviceEnablePeerAccess(peerDeviceId,flags) nogil:
    """@brief Enable direct access from current device's virtual address space to memory allocations
    physically located on a peer device.
    Memory which already allocated on peer device will be mapped into the address space of the
    current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
    the address space of the current device when the memory is allocated. The peer memory remains
    accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
    @param [in] peerDeviceId
    @param [in] flags
    Returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
    @returns #hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device.
    """
    pass

def hipDeviceDisablePeerAccess(peerDeviceId) nogil:
    """@brief Disable direct access from current device's virtual address space to memory allocations
    physically located on a peer device.
    Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
    enabled from the current device.
    @param [in] peerDeviceId
    @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
    """
    pass

def hipMemGetAddressRange(pbase,psize,dptr) nogil:
    """@brief Get information on memory allocations.
    @param [out] pbase - BAse pointer address
    @param [out] psize - Size of allocation
    @param [in]  dptr- Device Pointer
    @returns #hipSuccess, #hipErrorInvalidDevicePointer
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipMemcpyPeer(dst,dstDeviceId,src,srcDeviceId,sizeBytes) nogil:
    """@brief Copies memory from one device to memory on another device.
    @param [out] dst - Destination device pointer.
    @param [in] dstDeviceId - Destination device
    @param [in] src - Source device pointer
    @param [in] srcDeviceId - Source device
    @param [in] sizeBytes - Size of memory copy in bytes
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
    """
    pass

def hipMemcpyPeerAsync(dst,dstDeviceId,src,srcDevice,sizeBytes,stream) nogil:
    """@brief Copies memory from one device to memory on another device.
    @param [out] dst - Destination device pointer.
    @param [in] dstDevice - Destination device
    @param [in] src - Source device pointer
    @param [in] srcDevice - Source device
    @param [in] sizeBytes - Size of memory copy in bytes
    @param [in] stream - Stream identifier
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
    """
    pass

def hipCtxCreate(ctx,flags,device) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Context Context Management
    @{
    This section describes the context management functions of HIP runtime API.
    @addtogroup ContextD Context Management [Deprecated]
    @{
    @ingroup Context
    This section describes the deprecated context management functions of HIP runtime API.
    @brief Create a context and set it as current/ default context
    @param [out] ctx
    @param [in] flags
    @param [in] associated device handle
    @return #hipSuccess
    @see hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent,
    hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxDestroy(ctx) nogil:
    """@brief Destroy a HIP context.
    @param [in] ctx Context to destroy
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipCtxCreate, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,hipCtxSetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    """
    pass

def hipCtxPopCurrent(ctx) nogil:
    """@brief Pop the current/default context and return the popped context.
    @param [out] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxSetCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxPushCurrent(ctx) nogil:
    """@brief Push the context to be set as current/ default context
    @param [in] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    """
    pass

def hipCtxSetCurrent(ctx) nogil:
    """@brief Set the passed context as current/default
    @param [in] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    """
    pass

def hipCtxGetCurrent(ctx) nogil:
    """@brief Get the handle of the current/ default context
    @param [out] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxGetDevice(device) nogil:
    """@brief Get the handle of the device associated with current/default context
    @param [out] device
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize
    """
    pass

def hipCtxGetApiVersion(ctx,apiVersion) nogil:
    """@brief Returns the approximate HIP api version.
    @param [in]  ctx Context to check
    @param [out] apiVersion
    @return #hipSuccess
    @warning The HIP feature set does not correspond to an exact CUDA SDK api revision.
    This function always set *apiVersion to 4 as an approximation though HIP supports
    some features which were introduced in later CUDA SDK revisions.
    HIP apps code should not rely on the api revision number here and should
    use arch feature flags to test device capabilities or conditional compilation.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxGetCacheConfig(cacheConfig) nogil:
    """@brief Set Cache configuration for a specific function
    @param [out] cacheConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxSetCacheConfig(cacheConfig) nogil:
    """@brief Set L1/Shared cache partition.
    @param [in] cacheConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxSetSharedMemConfig(config) nogil:
    """@brief Set Shared memory bank configuration.
    @param [in] sharedMemoryConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxGetSharedMemConfig(pConfig) nogil:
    """@brief Get Shared memory bank configuration.
    @param [out] sharedMemoryConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxSynchronize() nogil:
    """@brief Blocks until the default context has completed all preceding requested tasks.
    @return #hipSuccess
    @warning This function waits for all streams on the default context to complete execution, and
    then returns.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxGetDevice
    """
    pass

def hipCtxGetFlags(flags) nogil:
    """@brief Return flags used for creating default context.
    @param [out] flags
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxEnablePeerAccess(peerCtx,flags) nogil:
    """@brief Enables direct access to memory allocations in a peer context.
    Memory which already allocated on peer device will be mapped into the address space of the
    current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
    the address space of the current device when the memory is allocated. The peer memory remains
    accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
    @param [in] peerCtx
    @param [in] flags
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
    #hipErrorPeerAccessAlreadyEnabled
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    @warning PeerToPeer support is experimental.
    """
    pass

def hipCtxDisablePeerAccess(peerCtx) nogil:
    """@brief Disable direct access from current context's virtual address space to memory allocations
    physically located on a peer context.Disables direct access to memory allocations in a peer
    context and unregisters any registered allocations.
    Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
    enabled from the current device.
    @param [in] peerCtx
    @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    @warning PeerToPeer support is experimental.
    """
    pass

def hipDevicePrimaryCtxGetState(dev,flags,active) nogil:
    """@}
    @brief Get the state of the primary context.
    @param [in] Device to get primary context flags for
    @param [out] Pointer to store flags
    @param [out] Pointer to store context state; 0 = inactive, 1 = active
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipDevicePrimaryCtxRelease(dev) nogil:
    """@brief Release the primary context on the GPU.
    @param [in] Device which primary context is released
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    @warning This function return #hipSuccess though doesn't release the primaryCtx by design on
    HIP/HCC path.
    """
    pass

def hipDevicePrimaryCtxRetain(pctx,dev) nogil:
    """@brief Retain the primary context on the GPU.
    @param [out] Returned context handle of the new context
    @param [in] Device which primary context is released
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipDevicePrimaryCtxReset(dev) nogil:
    """@brief Resets the primary context on the GPU.
    @param [in] Device which primary context is reset
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipDevicePrimaryCtxSetFlags(dev,flags) nogil:
    """@brief Set flags for the primary context.
    @param [in] Device for which the primary context flags are set
    @param [in] New flags for the device
    @returns #hipSuccess, #hipErrorContextAlreadyInUse
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipModuleLoad(module,fname) nogil:
    """@}
    @defgroup Module Module Management
    @{
    This section describes the module management functions of HIP runtime API.
    @brief Loads code object from file into a hipModule_t
    @param [in] fname
    @param [out] module
    @warning File/memory resources allocated in this function are released only in hipModuleUnload.
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorFileNotFound,
    hipErrorOutOfMemory, hipErrorSharedObjectInitFailed, hipErrorNotInitialized
    """
    pass

def hipModuleUnload(module) nogil:
    """@brief Frees the module
    @param [in] module
    @returns hipSuccess, hipInvalidValue
    module is freed and the code objects associated with it are destroyed
    """
    pass

def hipModuleGetFunction(function,module,kname) nogil:
    """@brief Function with kname will be extracted if present in module
    @param [in] module
    @param [in] kname
    @param [out] function
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorNotInitialized,
    hipErrorNotFound,
    """
    pass

def hipFuncGetAttributes(attr,func) nogil:
    """@brief Find out attributes for a given function.
    @param [out] attr
    @param [in] func
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
    """
    pass

def hipFuncGetAttribute(value,attrib,hfunc) nogil:
    """@brief Find out a specific attribute for a given function.
    @param [out] value
    @param [in]  attrib
    @param [in]  hfunc
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
    """
    pass

def hipModuleGetTexRef(texRef,hmod,name) nogil:
    """@brief returns the handle of the texture reference with the name from the module.
    @param [in] hmod
    @param [in] name
    @param [out] texRef
    @returns hipSuccess, hipErrorNotInitialized, hipErrorNotFound, hipErrorInvalidValue
    """
    pass

def hipModuleLoadData(module,image) nogil:
    """@brief builds module from code object which resides in host memory. Image is pointer to that
    location.
    @param [in] image
    @param [out] module
    @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
    """
    pass

def hipModuleLoadDataEx(module,image,numOptions,options,optionValues) nogil:
    """@brief builds module from code object which resides in host memory. Image is pointer to that
    location. Options are not used. hipModuleLoadData is called.
    @param [in] image
    @param [out] module
    @param [in] number of options
    @param [in] options for JIT
    @param [in] option values for JIT
    @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
    """
    pass

def hipModuleLaunchKernel(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,stream,kernelParams,extra) nogil:
    """@brief launches kernel f with launch parameters and shared memory on stream with arguments passed
    to kernelparams or extra
    @param [in] f         Kernel to launch.
    @param [in] gridDimX  X grid dimension specified as multiple of blockDimX.
    @param [in] gridDimY  Y grid dimension specified as multiple of blockDimY.
    @param [in] gridDimZ  Z grid dimension specified as multiple of blockDimZ.
    @param [in] blockDimX X block dimensions specified in work-items
    @param [in] blockDimY Y grid dimension specified in work-items
    @param [in] blockDimZ Z grid dimension specified in work-items
    @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
    default stream is used with associated synchronization rules.
    @param [in] kernelParams
    @param [in] extra     Pointer to kernel arguments.   These are passed directly to the kernel and
    must be in the memory layout and alignment expected by the kernel.
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32. So gridDim.x * blockDim.x, gridDim.y * blockDim.y
    and gridDim.z * blockDim.z are always less than 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    @warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please
    refer to hip_porting_driver_api.md for sample usage.
    """
    pass

def hipLaunchCooperativeKernel(f,gridDim,blockDimX,kernelParams,sharedMemBytes,stream) nogil:
    """@brief launches kernel f with launch parameters and shared memory on stream with arguments passed
    to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute
    @param [in] f         Kernel to launch.
    @param [in] gridDim   Grid dimensions specified as multiple of blockDim.
    @param [in] blockDim  Block dimensions specified in work-items
    @param [in] kernelParams A list of kernel arguments
    @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
    default stream is used with associated synchronization rules.
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
    """
    pass

def hipLaunchCooperativeKernelMultiDevice(launchParamsList,numDevices,flags) nogil:
    """@brief Launches kernels on multiple devices where thread blocks can cooperate and
    synchronize as they execute.
    @param [in] launchParamsList         List of launch parameters, one per device.
    @param [in] numDevices               Size of the launchParamsList array.
    @param [in] flags                    Flags to control launch behavior.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
    """
    pass

def hipExtLaunchMultiKernelMultiDevice(launchParamsList,numDevices,flags) nogil:
    """@brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
    on respective streams before enqueuing any other work on the specified streams from any other threads
    @param [in] hipLaunchParams          List of launch parameters, one per device.
    @param [in] numDevices               Size of the launchParamsList array.
    @param [in] flags                    Flags to control launch behavior.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    """
    pass

def hipModuleOccupancyMaxPotentialBlockSize(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit) nogil:
    """@}
    @defgroup Occupancy Occupancy
    @{
    This section describes the occupancy functions of HIP runtime API.
    @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
    @param [out] gridSize           minimum grid size for maximum potential occupancy
    @param [out] blockSize          block size for maximum potential occupancy
    @param [in]  f                  kernel function for which occupancy is calulated
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
    """
    pass

def hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit,flags) nogil:
    """@brief determine the grid and block sizes to achieves maximum occupancy for a kernel
    @param [out] gridSize           minimum grid size for maximum potential occupancy
    @param [out] blockSize          block size for maximum potential occupancy
    @param [in]  f                  kernel function for which occupancy is calulated
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
    @param [in]  flags            Extra flags for occupancy calculation (only default supported)
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
    """
    pass

def hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks,f,blockSize,dynSharedMemPerBlk) nogil:
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  func             Kernel function (hipFunction) for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    """
    pass

def hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks,f,blockSize,dynSharedMemPerBlk,flags) nogil:
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  f                Kernel function(hipFunction_t) for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  flags            Extra flags for occupancy calculation (only default supported)
    """
    pass

def hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks,f,blockSize,dynSharedMemPerBlk) nogil:
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  func             Kernel function for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    """
    pass

def hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks,f,blockSize,dynSharedMemPerBlk,flags) nogil:
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  f                Kernel function for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  flags            Extra flags for occupancy calculation (currently ignored)
    """
    pass

def hipOccupancyMaxPotentialBlockSize(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit) nogil:
    """@brief determine the grid and block sizes to achieves maximum occupancy for a kernel
    @param [out] gridSize           minimum grid size for maximum potential occupancy
    @param [out] blockSize          block size for maximum potential occupancy
    @param [in]  f                  kernel function for which occupancy is calulated
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
    """
    pass

def hipProfilerStart() nogil:
    """@brief Start recording of profiling information
    When using this API, start the profiler with profiling disabled.  (--startdisabled)
    @warning : hipProfilerStart API is under development.
    """
    pass

def hipProfilerStop() nogil:
    """@brief Stop recording of profiling information.
    When using this API, start the profiler with profiling disabled.  (--startdisabled)
    @warning : hipProfilerStop API is under development.
    """
    pass

def hipConfigureCall(gridDim,blockDim,sharedMem,stream) nogil:
    """@}
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    @defgroup Clang Launch API to support the triple-chevron syntax
    @{
    This section describes the API to support the triple-chevron syntax.
    @brief Configure a kernel launch.
    @param [in] gridDim   grid dimension specified as multiple of blockDim.
    @param [in] blockDim  block dimensions specified in work-items
    @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
    default stream is used with associated synchronization rules.
    Please note, HIP does not support kernel launch with total work items defined in dimension with
    size gridDim x blockDim >= 2^32.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    """
    pass

def hipSetupArgument(arg,size,offset) nogil:
    """@brief Set a kernel argument.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    @param [in] arg    Pointer the argument in host memory.
    @param [in] size   Size of the argument.
    @param [in] offset Offset of the argument on the argument stack.
    """
    pass

def hipLaunchByPtr(func) nogil:
    """@brief Launch a kernel.
    @param [in] func Kernel to launch.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    """
    pass

def hipLaunchKernel(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream) nogil:
    """@brief C compliant kernel launch API
    @param [in] function_address - kernel stub function pointer.
    @param [in] numBlocks - number of blocks
    @param [in] dimBlocks - dimension of a block
    @param [in] args - kernel arguments
    @param [in] sharedMemBytes - Amount of dynamic shared memory to allocate for this kernel. The
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream - Stream where the kernel should be dispatched.  May be 0, in which case th
    default stream is used with associated synchronization rules.
    @returns #hipSuccess, #hipErrorInvalidValue, hipInvalidDevice
    """
    pass

def hipLaunchHostFunc(stream,fn,userData) nogil:
    """@brief Enqueues a host function call in a stream.
    @param [in] stream - stream to enqueue work to.
    @param [in] fn - function to call once operations enqueued preceeding are complete.
    @param [in] userData - User-specified data to be passed to the function.
    @returns #hipSuccess, #hipErrorInvalidResourceHandle, #hipErrorInvalidValue,
    #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDrvMemcpy2DUnaligned(pCopy) nogil:
    """Copies memory for 2D arrays.
    @param pCopy           - Parameters for the memory copy
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipExtLaunchKernel(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream,startEvent,stopEvent,flags) nogil:
    """@brief Launches kernel from the pointer address, with arguments and shared memory on stream.
    @param [in] function_address pointer to the Kernel to launch.
    @param [in] numBlocks number of blocks.
    @param [in] dimBlocks dimension of a block.
    @param [in] args pointer to kernel arguments.
    @param [in] sharedMemBytes  Amount of dynamic shared memory to allocate for this kernel.
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream  Stream where the kernel should be dispatched.
    @param [in] startEvent  If non-null, specified event will be updated to track the start time of
    the kernel launch. The event must be created before calling this API.
    @param [in] stopEvent  If non-null, specified event will be updated to track the stop time of
    the kernel launch. The event must be created before calling this API.
    May be 0, in which case the default stream is used with associated synchronization rules.
    @param [in] flags. The value of hipExtAnyOrderLaunch, signifies if kernel can be
    launched in any order.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue.
    """
    pass

def hipBindTextureToMipmappedArray(tex,mipmappedArray,desc) nogil:
    """@brief  Binds a mipmapped array to a texture.
    @param [in] tex  pointer to the texture reference to bind
    @param [in] mipmappedArray  memory mipmapped array on the device
    @param [in] desc  opointer to the channel format
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipCreateTextureObject(pTexObject,pResDesc,pTexDesc,pResViewDesc) nogil:
    """@brief Creates a texture object.
    @param [out] pTexObject  pointer to the texture object to create
    @param [in] pResDesc  pointer to resource descriptor
    @param [in] pTexDesc  pointer to texture descriptor
    @param [in] pResViewDesc  pointer to resource view descriptor
    @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported, hipErrorOutOfMemory
    @note 3D liner filter isn't supported on GFX90A boards, on which the API @p hipCreateTextureObject will
    return hipErrorNotSupported.
    """
    pass

def hipDestroyTextureObject(textureObject) nogil:
    """@brief Destroys a texture object.
    @param [in] textureObject  texture object to destroy
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetChannelDesc(desc,array) nogil:
    """@brief Gets the channel descriptor in an array.
    @param [in] desc  pointer to channel format descriptor
    @param [out] array  memory array on the device
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetTextureObjectResourceDesc(pResDesc,textureObject) nogil:
    """@brief Gets resource descriptor for the texture object.
    @param [out] pResDesc  pointer to resource descriptor
    @param [in] textureObject  texture object
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetTextureObjectResourceViewDesc(pResViewDesc,textureObject) nogil:
    """@brief Gets resource view descriptor for the texture object.
    @param [out] pResViewDesc  pointer to resource view descriptor
    @param [in] textureObject  texture object
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetTextureObjectTextureDesc(pTexDesc,textureObject) nogil:
    """@brief Gets texture descriptor for the texture object.
    @param [out] pTexDesc  pointer to texture descriptor
    @param [in] textureObject  texture object
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexObjectCreate(pTexObject,pResDesc,pTexDesc,pResViewDesc) nogil:
    """@brief Creates a texture object.
    @param [out] pTexObject  pointer to texture object to create
    @param [in] pResDesc  pointer to resource descriptor
    @param [in] pTexDesc  pointer to texture descriptor
    @param [in] pResViewDesc  pointer to resource view descriptor
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexObjectDestroy(texObject) nogil:
    """@brief Destroys a texture object.
    @param [in] texObject  texture object to destroy
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexObjectGetResourceDesc(pResDesc,texObject) nogil:
    """@brief Gets resource descriptor of a texture object.
    @param [out] pResDesc  pointer to resource descriptor
    @param [in] texObject  texture object
    @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    """
    pass

def hipTexObjectGetResourceViewDesc(pResViewDesc,texObject) nogil:
    """@brief Gets resource view descriptor of a texture object.
    @param [out] pResViewDesc  pointer to resource view descriptor
    @param [in] texObject  texture object
    @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    """
    pass

def hipTexObjectGetTextureDesc(pTexDesc,texObject) nogil:
    """@brief Gets texture descriptor of a texture object.
    @param [out] pTexDesc  pointer to texture descriptor
    @param [in] texObject  texture object
    @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    """
    pass

def hipGetTextureReference(texref,symbol) nogil:
    """@addtogroup TextureD Texture Management [Deprecated]
    @{
    @ingroup Texture
    This section describes the deprecated texture management functions of HIP runtime API.
    @brief Gets the texture reference related with the symbol.
    @param [out] texref  texture reference
    @param [in] symbol  pointer to the symbol related with the texture for the reference
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexRefSetAddressMode(texRef,dim,am) nogil:
    """
    """
    pass

def hipTexRefSetArray(tex,array,flags) nogil:
    """
    """
    pass

def hipTexRefSetFilterMode(texRef,fm) nogil:
    """
    """
    pass

def hipTexRefSetFlags(texRef,Flags) nogil:
    """
    """
    pass

def hipTexRefSetFormat(texRef,fmt,NumPackedComponents) nogil:
    """
    """
    pass

def hipBindTexture(offset,tex,devPtr,desc,size) nogil:
    """
    """
    pass

def hipBindTexture2D(offset,tex,devPtr,desc,width,height,pitch) nogil:
    """
    """
    pass

def hipBindTextureToArray(tex,array,desc) nogil:
    """
    """
    pass

def hipGetTextureAlignmentOffset(offset,texref) nogil:
    """
    """
    pass

def hipUnbindTexture(tex) nogil:
    """
    """
    pass

def hipTexRefGetAddress(dev_ptr,texRef) nogil:
    """
    """
    pass

def hipTexRefGetAddressMode(pam,texRef,dim) nogil:
    """
    """
    pass

def hipTexRefGetFilterMode(pfm,texRef) nogil:
    """
    """
    pass

def hipTexRefGetFlags(pFlags,texRef) nogil:
    """
    """
    pass

def hipTexRefGetFormat(pFormat,pNumChannels,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMaxAnisotropy(pmaxAnsio,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMipmapFilterMode(pfm,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMipmapLevelBias(pbias,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp,pmaxMipmapLevelClamp,texRef) nogil:
    """
    """
    pass

def hipTexRefGetMipMappedArray(pArray,texRef) nogil:
    """
    """
    pass

def hipTexRefSetAddress(ByteOffset,texRef,dptr,bytes) nogil:
    """
    """
    pass

def hipTexRefSetAddress2D(texRef,desc,dptr,Pitch) nogil:
    """
    """
    pass

def hipTexRefSetMaxAnisotropy(texRef,maxAniso) nogil:
    """
    """
    pass

def hipTexRefSetBorderColor(texRef,pBorderColor) nogil:
    """
    """
    pass

def hipTexRefSetMipmapFilterMode(texRef,fm) nogil:
    """
    """
    pass

def hipTexRefSetMipmapLevelBias(texRef,bias) nogil:
    """
    """
    pass

def hipTexRefSetMipmapLevelClamp(texRef,minMipMapLevelClamp,maxMipMapLevelClamp) nogil:
    """
    """
    pass

def hipTexRefSetMipmappedArray(texRef,mipmappedArray,Flags) nogil:
    """
    """
    pass

def hipMipmappedArrayCreate(pHandle,pMipmappedArrayDesc,numMipmapLevels) nogil:
    """@addtogroup TextureU Texture Management [Not supported]
    @{
    @ingroup Texture
    This section describes the texture management functions currently unsupported in HIP runtime.
    """
    pass

def hipMipmappedArrayDestroy(hMipmappedArray) nogil:
    """
    """
    pass

def hipMipmappedArrayGetLevel(pLevelArray,hMipMappedArray,level) nogil:
    """
    """
    pass

def hipApiName(id) nogil:
    """@defgroup Callback Callback Activity APIs
    @{
    This section describes the callback/Activity of HIP runtime API.
    """
    pass

def hipKernelNameRef(f) nogil:
    """
    """
    pass

def hipKernelNameRefByPtr(hostFunction,stream) nogil:
    """
    """
    pass

def hipGetStreamDeviceId(stream) nogil:
    """
    """
    pass

def hipStreamBeginCapture(stream,mode) nogil:
    """@brief Begins graph capture on a stream.
    @param [in] stream - Stream to initiate capture.
    @param [in] mode - Controls the interaction of this capture sequence with other API calls that
    are not safe.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamEndCapture(stream,pGraph) nogil:
    """@brief Ends capture on a stream, returning the captured graph.
    @param [in] stream - Stream to end capture.
    @param [out] pGraph - returns the graph captured.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamGetCaptureInfo(stream,pCaptureStatus,pId) nogil:
    """@brief Get capture status of a stream.
    @param [in] stream - Stream under capture.
    @param [out] pCaptureStatus - returns current status of the capture.
    @param [out] pId - unique ID of the capture.
    @returns #hipSuccess, #hipErrorStreamCaptureImplicit
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamGetCaptureInfo_v2(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out) nogil:
    """@brief Get stream's capture state
    @param [in] stream - Stream under capture.
    @param [out] captureStatus_out - returns current status of the capture.
    @param [out] id_out - unique ID of the capture.
    @param [in] graph_out - returns the graph being captured into.
    @param [out] dependencies_out - returns pointer to an array of nodes.
    @param [out] numDependencies_out - returns size of the array returned in dependencies_out.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamIsCapturing(stream,pCaptureStatus) nogil:
    """@brief Get stream's capture state
    @param [in] stream - Stream under capture.
    @param [out] pCaptureStatus - returns current status of the capture.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamUpdateCaptureDependencies(stream,dependencies,numDependencies,flags) nogil:
    """@brief Update the set of dependencies in a capturing stream
    @param [in] stream - Stream under capture.
    @param [in] dependencies - pointer to an array of nodes to Add/Replace.
    @param [in] numDependencies - size of the array in dependencies.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorIllegalState
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipThreadExchangeStreamCaptureMode(mode) nogil:
    """@brief Swaps the stream capture mode of a thread.
    @param [in] mode - Pointer to mode value to swap with the current mode
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphCreate(pGraph,flags) nogil:
    """@brief Creates a graph
    @param [out] pGraph - pointer to graph to create.
    @param [in] flags - flags for graph creation, must be 0.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphDestroy(graph) nogil:
    """@brief Destroys a graph
    @param [in] graph - instance of graph to destroy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddDependencies(graph,from,to,numDependencies) nogil:
    """@brief Adds dependency edges to a graph.
    @param [in] graph - instance of the graph to add dependencies.
    @param [in] from - pointer to the graph nodes with dependenties to add from.
    @param [in] to - pointer to the graph nodes to add dependenties to.
    @param [in] numDependencies - the number of dependencies to add.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphRemoveDependencies(graph,from,to,numDependencies) nogil:
    """@brief Removes dependency edges from a graph.
    @param [in] graph - instance of the graph to remove dependencies.
    @param [in] from - Array of nodes that provide the dependencies.
    @param [in] to - Array of dependent nodes.
    @param [in] numDependencies - the number of dependencies to remove.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphGetEdges(graph,from,to,numEdges) nogil:
    """@brief Returns a graph's dependency edges.
    @param [in] graph - instance of the graph to get the edges from.
    @param [out] from - pointer to the graph nodes to return edge endpoints.
    @param [out] to - pointer to the graph nodes to return edge endpoints.
    @param [out] numEdges - returns number of edges.
    @returns #hipSuccess, #hipErrorInvalidValue
    from and to may both be NULL, in which case this function only returns the number of edges in
    numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher than the actual
    number of edges, the remaining entries in from and to will be set to NULL, and the number of
    edges actually returned will be written to numEdges
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphGetNodes(graph,nodes,numNodes) nogil:
    """@brief Returns graph nodes.
    @param [in] graph - instance of graph to get the nodes.
    @param [out] nodes - pointer to return the  graph nodes.
    @param [out] numNodes - returns number of graph nodes.
    @returns #hipSuccess, #hipErrorInvalidValue
    nodes may be NULL, in which case this function will return the number of nodes in numNodes.
    Otherwise, numNodes entries will be filled in. If numNodes is higher than the actual number of
    nodes, the remaining entries in nodes will be set to NULL, and the number of nodes actually
    obtained will be returned in numNodes.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphGetRootNodes(graph,pRootNodes,pNumRootNodes) nogil:
    """@brief Returns graph's root nodes.
    @param [in] graph - instance of the graph to get the nodes.
    @param [out] pRootNodes - pointer to return the graph's root nodes.
    @param [out] pNumRootNodes - returns the number of graph's root nodes.
    @returns #hipSuccess, #hipErrorInvalidValue
    pRootNodes may be NULL, in which case this function will return the number of root nodes in
    pNumRootNodes. Otherwise, pNumRootNodes entries will be filled in. If pNumRootNodes is higher
    than the actual number of root nodes, the remaining entries in pRootNodes will be set to NULL,
    and the number of nodes actually obtained will be returned in pNumRootNodes.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphNodeGetDependencies(node,pDependencies,pNumDependencies) nogil:
    """@brief Returns a node's dependencies.
    @param [in] node - graph node to get the dependencies from.
    @param [out] pDependencies - pointer to to return the dependencies.
    @param [out] pNumDependencies -  returns the number of graph node dependencies.
    @returns #hipSuccess, #hipErrorInvalidValue
    pDependencies may be NULL, in which case this function will return the number of dependencies in
    pNumDependencies. Otherwise, pNumDependencies entries will be filled in. If pNumDependencies is
    higher than the actual number of dependencies, the remaining entries in pDependencies will be set
    to NULL, and the number of nodes actually obtained will be returned in pNumDependencies.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphNodeGetDependentNodes(node,pDependentNodes,pNumDependentNodes) nogil:
    """@brief Returns a node's dependent nodes.
    @param [in] node - graph node to get the Dependent nodes from.
    @param [out] pDependentNodes - pointer to return the graph dependent nodes.
    @param [out] pNumDependentNodes - returns the number of graph node dependent nodes.
    @returns #hipSuccess, #hipErrorInvalidValue
    DependentNodes may be NULL, in which case this function will return the number of dependent nodes
    in pNumDependentNodes. Otherwise, pNumDependentNodes entries will be filled in. If
    pNumDependentNodes is higher than the actual number of dependent nodes, the remaining entries in
    pDependentNodes will be set to NULL, and the number of nodes actually obtained will be returned
    in pNumDependentNodes.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphNodeGetType(node,pType) nogil:
    """@brief Returns a node's type.
    @param [in] node - instance of the graph to add dependencies.
    @param [out] pType - pointer to the return the type
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphDestroyNode(node) nogil:
    """@brief Remove a node from the graph.
    @param [in] node - graph node to remove
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphClone(pGraphClone,originalGraph) nogil:
    """@brief Clones a graph.
    @param [out] pGraphClone - Returns newly created cloned graph.
    @param [in] originalGraph - original graph to clone from.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphNodeFindInClone(pNode,originalNode,clonedGraph) nogil:
    """@brief Finds a cloned version of a node.
    @param [out] pNode - Returns the cloned node.
    @param [in] originalNode - original node handle.
    @param [in] clonedGraph - Cloned graph to query.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphInstantiate(pGraphExec,graph,pErrorNode,pLogBuffer,bufferSize) nogil:
    """@brief Creates an executable graph from a graph
    @param [out] pGraphExec - pointer to instantiated executable graph that is created.
    @param [in] graph - instance of graph to instantiate.
    @param [out] pErrorNode - pointer to error node in case error occured in graph instantiation,
    it could modify the correponding node.
    @param [out] pLogBuffer - pointer to log buffer.
    @param [out] bufferSize - the size of log buffer.
    @returns #hipSuccess, #hipErrorOutOfMemory
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphInstantiateWithFlags(pGraphExec,graph,flags) nogil:
    """@brief Creates an executable graph from a graph.
    @param [out] pGraphExec - pointer to instantiated executable graph that is created.
    @param [in] graph - instance of graph to instantiate.
    @param [in] flags - Flags to control instantiation.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphLaunch(graphExec,stream) nogil:
    """@brief launches an executable graph in a stream
    @param [in] graphExec - instance of executable graph to launch.
    @param [in] stream - instance of stream in which to launch executable graph.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphUpload(graphExec,stream) nogil:
    """@brief uploads an executable graph in a stream
    @param [in] graphExec - instance of executable graph to launch.
    @param [in] stream - instance of stream in which to launch executable graph.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecDestroy(graphExec) nogil:
    """@brief Destroys an executable graph
    @param [in] pGraphExec - instance of executable graph to destry.
    @returns #hipSuccess.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecUpdate(hGraphExec,hGraph,hErrorNode_out,updateResult_out) nogil:
    """@brief Check whether an executable graph can be updated with a graph and perform the update if  *
    possible.
    @param [in] hGraphExec - instance of executable graph to update.
    @param [in] hGraph - graph that contains the updated parameters.
    @param [in] hErrorNode_out -  node which caused the permissibility check to forbid the update.
    @param [in] updateResult_out - Whether the graph update was permitted.
    @returns #hipSuccess, #hipErrorGraphExecUpdateFailure
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddKernelNode(pGraphNode,graph,pDependencies,numDependencies,pNodeParams) nogil:
    """@brief Creates a kernel execution node and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - pointer to the dependencies on the kernel execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] pNodeParams - pointer to the parameters to the kernel execution node on the GPU.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDeviceFunction
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeGetParams(node,pNodeParams) nogil:
    """@brief Gets kernel node's parameters.
    @param [in] node - instance of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeSetParams(node,pNodeParams) nogil:
    """@brief Sets a kernel node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecKernelNodeSetParams(hGraphExec,node,pNodeParams) nogil:
    """@brief Sets the parameters for a kernel node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the kernel node parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNode(pGraphNode,graph,pDependencies,numDependencies,pCopyParams) nogil:
    """@brief Creates a memcpy node and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] pCopyParams - const pointer to the parameters for the memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeGetParams(node,pNodeParams) nogil:
    """@brief Gets a memcpy node's parameters.
    @param [in] node - instance of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeSetParams(node,pNodeParams) nogil:
    """@brief Sets a memcpy node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeSetAttribute(hNode,attr,value) nogil:
    """@brief Sets a node attribute.
    @param [in] hNode - instance of the node to set parameters to.
    @param [in] attr - the attribute node is set to.
    @param [in] value - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeGetAttribute(hNode,attr,value) nogil:
    """@brief Gets a node attribute.
    @param [in] hNode - instance of the node to set parameters to.
    @param [in] attr - the attribute node is set to.
    @param [in] value - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemcpyNodeSetParams(hGraphExec,node,pNodeParams) nogil:
    """@brief Sets the parameters for a memcpy node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the kernel node parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNode1D(pGraphNode,graph,pDependencies,numDependencies,dst,src,count,kind) nogil:
    """@brief Creates a 1D memcpy node and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] src - pointer to memory address to the source.
    @param [in] count - the size of the memory to copy.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeSetParams1D(node,dst,src,count,kind) nogil:
    """@brief Sets a memcpy node's parameters to perform a 1-dimensional copy.
    @param [in] node - instance of the node to set parameters to.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] src - pointer to memory address to the source.
    @param [in] count - the size of the memory to copy.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemcpyNodeSetParams1D(hGraphExec,node,dst,src,count,kind) nogil:
    """@brief Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional
    copy.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] src - pointer to memory address to the source.
    @param [in] count - the size of the memory to copy.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNodeFromSymbol(pGraphNode,graph,pDependencies,numDependencies,dst,symbol,count,offset,kind) nogil:
    """@brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] symbol - Device symbol address.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeSetParamsFromSymbol(node,dst,symbol,count,offset,kind) nogil:
    """@brief Sets a memcpy node's parameters to copy from a symbol on the device.
    @param [in] node - instance of the node to set parameters to.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] symbol - Device symbol address.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec,node,dst,symbol,count,offset,kind) nogil:
    """@brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the
    device.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] dst - pointer to memory address to the destination.
    @param [in] symbol - Device symbol address.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNodeToSymbol(pGraphNode,graph,pDependencies,numDependencies,symbol,src,count,offset,kind) nogil:
    """@brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph.
    @param [out] pGraphNode - pointer to graph node to create.
    @param [in] graph - instance of graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] symbol - Device symbol address.
    @param [in] src - pointer to memory address of the src.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeSetParamsToSymbol(node,symbol,src,count,offset,kind) nogil:
    """@brief Sets a memcpy node's parameters to copy to a symbol on the device.
    @param [in] node - instance of the node to set parameters to.
    @param [in] symbol - Device symbol address.
    @param [in] src - pointer to memory address of the src.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec,node,symbol,src,count,offset,kind) nogil:
    """@brief Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the
    device.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] symbol - Device symbol address.
    @param [in] src - pointer to memory address of the src.
    @param [in] count - the size of the memory to copy.
    @param [in] offset - Offset from start of symbol in bytes.
    @param [in] kind - the type of memory copy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemsetNode(pGraphNode,graph,pDependencies,numDependencies,pMemsetParams) nogil:
    """@brief Creates a memset node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create.
    @param [in] graph - instance of the graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] pMemsetParams - const pointer to the parameters for the memory set.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemsetNodeGetParams(node,pNodeParams) nogil:
    """@brief Gets a memset node's parameters.
    @param [in] node - instane of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemsetNodeSetParams(node,pNodeParams) nogil:
    """@brief Sets a memset node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemsetNodeSetParams(hGraphExec,node,pNodeParams) nogil:
    """@brief Sets the parameters for a memset node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddHostNode(pGraphNode,graph,pDependencies,numDependencies,pNodeParams) nogil:
    """@brief Creates a host execution node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create.
    @param [in] graph - instance of the graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] pNodeParams -pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphHostNodeGetParams(node,pNodeParams) nogil:
    """@brief Returns a host node's parameters.
    @param [in] node - instane of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphHostNodeSetParams(node,pNodeParams) nogil:
    """@brief Sets a host node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecHostNodeSetParams(hGraphExec,node,pNodeParams) nogil:
    """@brief Sets the parameters for a host node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddChildGraphNode(pGraphNode,graph,pDependencies,numDependencies,childGraph) nogil:
    """@brief Creates a child graph node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create.
    @param [in] graph - instance of the graph to add the created node.
    @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
    @param [in] numDependencies - the number of the dependencies.
    @param [in] childGraph - the graph to clone into this node
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphChildGraphNodeGetGraph(node,pGraph) nogil:
    """@brief Gets a handle to the embedded graph of a child graph node.
    @param [in] node - instane of the node to get child graph.
    @param [out] pGraph - pointer to get the graph.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecChildGraphNodeSetParams(hGraphExec,node,childGraph) nogil:
    """@brief Updates node parameters in the child graph node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - node from the graph which was used to instantiate graphExec.
    @param [in] childGraph - child graph with updated parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddEmptyNode(pGraphNode,graph,pDependencies,numDependencies) nogil:
    """@brief Creates an empty node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    @param [in] graph - instane of the graph the node is add to.
    @param [in] pDependencies - const pointer to the node dependenties.
    @param [in] numDependencies - the number of dependencies.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddEventRecordNode(pGraphNode,graph,pDependencies,numDependencies,event) nogil:
    """@brief Creates an event record node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    @param [in] graph - instane of the graph the node to be added.
    @param [in] pDependencies - const pointer to the node dependenties.
    @param [in] numDependencies - the number of dependencies.
    @param [in] event - Event for the node.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphEventRecordNodeGetEvent(node,event_out) nogil:
    """@brief Returns the event associated with an event record node.
    @param [in] node -  instane of the node to get event from.
    @param [out] event_out - Pointer to return the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphEventRecordNodeSetEvent(node,event) nogil:
    """@brief Sets an event record node's event.
    @param [in] node - instane of the node to set event to.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecEventRecordNodeSetEvent(hGraphExec,hNode,event) nogil:
    """@brief Sets the event for an event record node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] hNode - node from the graph which was used to instantiate graphExec.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddEventWaitNode(pGraphNode,graph,pDependencies,numDependencies,event) nogil:
    """@brief Creates an event wait node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    @param [in] graph - instane of the graph the node to be added.
    @param [in] pDependencies - const pointer to the node dependenties.
    @param [in] numDependencies - the number of dependencies.
    @param [in] event - Event for the node.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphEventWaitNodeGetEvent(node,event_out) nogil:
    """@brief Returns the event associated with an event wait node.
    @param [in] node -  instane of the node to get event from.
    @param [out] event_out - Pointer to return the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphEventWaitNodeSetEvent(node,event) nogil:
    """@brief Sets an event wait node's event.
    @param [in] node - instane of the node to set event to.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecEventWaitNodeSetEvent(hGraphExec,hNode,event) nogil:
    """@brief Sets the event for an event record node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] hNode - node from the graph which was used to instantiate graphExec.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceGetGraphMemAttribute(device,attr,value) nogil:
    """@brief Get the mem attribute for graphs.
    @param [in] device - device the attr is get for.
    @param [in] attr - attr to get.
    @param [out] value - value for specific attr.
    @returns #hipSuccess, #hipErrorInvalidDevice
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceSetGraphMemAttribute(device,attr,value) nogil:
    """@brief Set the mem attribute for graphs.
    @param [in] device - device the attr is set for.
    @param [in] attr - attr to set.
    @param [in] value - value for specific attr.
    @returns #hipSuccess, #hipErrorInvalidDevice
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceGraphMemTrim(device) nogil:
    """@brief Free unused memory on specific device used for graph back to OS.
    @param [in] device - device the memory is used for graphs
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipUserObjectCreate(object_out,ptr,destroy,initialRefcount,flags) nogil:
    """@brief Create an instance of userObject to manage lifetime of a resource.
    @param [out] object_out - pointer to instace of userobj.
    @param [in] ptr - pointer to pass to destroy function.
    @param [in] destroy - destroy callback to remove resource.
    @param [in] initialRefcount - reference to resource.
    @param [in] flags - flags passed to API.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipUserObjectRelease(object,count) nogil:
    """@brief Release number of references to resource.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipUserObjectRetain(object,count) nogil:
    """@brief Retain number of references to resource.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphRetainUserObject(graph,object,count,flags) nogil:
    """@brief Retain user object for graphs.
    @param [in] graph - pointer to graph to retain the user object for.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @param [in] flags - flags passed to API.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphReleaseUserObject(graph,object,count) nogil:
    """@brief Release user object from graphs.
    @param [in] graph - pointer to graph to retain the user object for.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemAddressFree(devPtr,size) nogil:
    """@brief Frees an address range reservation made via hipMemAddressReserve
    @param [in] devPtr - starting address of the range.
    @param [in] size - size of the range.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemAddressReserve(ptr,size,alignment,addr,flags) nogil:
    """@brief Reserves an address range
    @param [out] ptr - starting address of the reserved range.
    @param [in] size - size of the reservation.
    @param [in] alignment - alignment of the address.
    @param [in] addr - requested starting address of the range.
    @param [in] flags - currently unused, must be zero.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemCreate(handle,size,prop,flags) nogil:
    """@brief Creates a memory allocation described by the properties and size
    @param [out] handle - value of the returned handle.
    @param [in] size - size of the allocation.
    @param [in] prop - properties of the allocation.
    @param [in] flags - currently unused, must be zero.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemExportToShareableHandle(shareableHandle,handle,handleType,flags) nogil:
    """@brief Exports an allocation to a requested shareable handle type.
    @param [out] shareableHandle - value of the returned handle.
    @param [in] handle - handle to share.
    @param [in] handleType - type of the shareable handle.
    @param [in] flags - currently unused, must be zero.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemGetAccess(flags,location,ptr) nogil:
    """@brief Get the access flags set for the given location and ptr.
    @param [out] flags - flags for this location.
    @param [in] location - target location.
    @param [in] ptr - address to check the access flags.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemGetAllocationGranularity(granularity,prop,option) nogil:
    """@brief Calculates either the minimal or recommended granularity.
    @param [out] granularity - returned granularity.
    @param [in] prop - location properties.
    @param [in] option - determines which granularity to return.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemGetAllocationPropertiesFromHandle(prop,handle) nogil:
    """@brief Retrieve the property structure of the given handle.
    @param [out] prop - properties of the given handle.
    @param [in] handle - handle to perform the query on.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemImportFromShareableHandle(handle,osHandle,shHandleType) nogil:
    """@brief Imports an allocation from a requested shareable handle type.
    @param [out] handle - returned value.
    @param [in] osHandle - shareable handle representing the memory allocation.
    @param [in] shHandleType - handle type.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemMap(ptr,size,offset,handle,flags) nogil:
    """@brief Maps an allocation handle to a reserved virtual address range.
    @param [in] ptr - address where the memory will be mapped.
    @param [in] size - size of the mapping.
    @param [in] offset - offset into the memory, currently must be zero.
    @param [in] handle - memory allocation to be mapped.
    @param [in] flags - currently unused, must be zero.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemMapArrayAsync(mapInfoList,count,stream) nogil:
    """@brief Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.
    @param [in] mapInfoList - list of hipArrayMapInfo.
    @param [in] count - number of hipArrayMapInfo in mapInfoList.
    @param [in] stream - stream identifier for the stream to use for map or unmap operations.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemRelease(handle) nogil:
    """@brief Release a memory handle representing a memory allocation which was previously allocated through hipMemCreate.
    @param [in] handle - handle of the memory allocation.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemRetainAllocationHandle(handle,addr) nogil:
    """@brief Returns the allocation handle of the backing memory allocation given the address.
    @param [out] handle - handle representing addr.
    @param [in] addr - address to look up.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemSetAccess(ptr,size,desc,count) nogil:
    """@brief Set the access flags for each location specified in desc for the given virtual address range.
    @param [in] ptr - starting address of the virtual address range.
    @param [in] size - size of the range.
    @param [in] desc - array of hipMemAccessDesc.
    @param [in] count - number of hipMemAccessDesc in desc.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemUnmap(ptr,size) nogil:
    """@brief Unmap memory allocation of a given address range.
    @param [in] ptr - starting address of the range to unmap.
    @param [in] size - size of the virtual address range.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGLGetDevices(pHipDeviceCount,pHipDevices,hipDeviceCount,deviceList) nogil:
    """
    """
    pass

def hipGraphicsGLRegisterBuffer(resource,buffer,flags) nogil:
    """
    """
    pass

def hipGraphicsGLRegisterImage(resource,image,target,flags) nogil:
    """
    """
    pass

def hipGraphicsMapResources(count,resources,stream) nogil:
    """
    """
    pass

def hipGraphicsSubResourceGetMappedArray(array,resource,arrayIndex,mipLevel) nogil:
    """
    """
    pass

def hipGraphicsResourceGetMappedPointer(devPtr,size,resource) nogil:
    """
    """
    pass

def hipGraphicsUnmapResources(count,resources,stream) nogil:
    """
    """
    pass

def hipGraphicsUnregisterResource(resource) nogil:
    """
    """
    pass

def hipMemcpy_spt(dst,src,sizeBytes,kind) nogil:
    """
    """
    pass

def hipMemcpyToSymbol_spt(symbol,src,sizeBytes,offset,kind) nogil:
    """
    """
    pass

def hipMemcpyFromSymbol_spt(dst,symbol,sizeBytes,offset,kind) nogil:
    """
    """
    pass

def hipMemcpy2D_spt(dst,dpitch,src,spitch,width,height,kind) nogil:
    """
    """
    pass

def hipMemcpy2DFromArray_spt(dst,dpitch,src,wOffset,hOffset,width,height,kind) nogil:
    """
    """
    pass

def hipMemcpy3D_spt(p) nogil:
    """
    """
    pass

def hipMemset_spt(dst,value,sizeBytes) nogil:
    """
    """
    pass

def hipMemsetAsync_spt(dst,value,sizeBytes,stream) nogil:
    """
    """
    pass

def hipMemset2D_spt(dst,pitch,value,width,height) nogil:
    """
    """
    pass

def hipMemset2DAsync_spt(dst,pitch,value,width,height,stream) nogil:
    """
    """
    pass

def hipMemset3DAsync_spt(pitchedDevPtr,value,extent,stream) nogil:
    """
    """
    pass

def hipMemset3D_spt(pitchedDevPtr,value,extent) nogil:
    """
    """
    pass

def hipMemcpyAsync_spt(dst,src,sizeBytes,kind,stream) nogil:
    """
    """
    pass

def hipMemcpy3DAsync_spt(p,stream) nogil:
    """
    """
    pass

def hipMemcpy2DAsync_spt(dst,dpitch,src,spitch,width,height,kind,stream) nogil:
    """
    """
    pass

def hipMemcpyFromSymbolAsync_spt(dst,symbol,sizeBytes,offset,kind,stream) nogil:
    """
    """
    pass

def hipMemcpyToSymbolAsync_spt(symbol,src,sizeBytes,offset,kind,stream) nogil:
    """
    """
    pass

def hipMemcpyFromArray_spt(dst,src,wOffsetSrc,hOffset,count,kind) nogil:
    """
    """
    pass

def hipMemcpy2DToArray_spt(dst,wOffset,hOffset,src,spitch,width,height,kind) nogil:
    """
    """
    pass

def hipMemcpy2DFromArrayAsync_spt(dst,dpitch,src,wOffsetSrc,hOffsetSrc,width,height,kind,stream) nogil:
    """
    """
    pass

def hipMemcpy2DToArrayAsync_spt(dst,wOffset,hOffset,src,spitch,width,height,kind,stream) nogil:
    """
    """
    pass

def hipStreamQuery_spt(stream) nogil:
    """
    """
    pass

def hipStreamSynchronize_spt(stream) nogil:
    """
    """
    pass

def hipStreamGetPriority_spt(stream,priority) nogil:
    """
    """
    pass

def hipStreamWaitEvent_spt(stream,event,flags) nogil:
    """
    """
    pass

def hipStreamGetFlags_spt(stream,flags) nogil:
    """
    """
    pass

def hipStreamAddCallback_spt(stream,callback,userData,flags) nogil:
    """
    """
    pass

def hipEventRecord_spt(event,stream) nogil:
    """
    """
    pass

def hipLaunchCooperativeKernel_spt(f,gridDim,blockDim,kernelParams,sharedMemBytes,hStream) nogil:
    """
    """
    pass

def hipLaunchKernel_spt(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream) nogil:
    """
    """
    pass

def hipGraphLaunch_spt(graphExec,stream) nogil:
    """
    """
    pass

def hipStreamBeginCapture_spt(stream,mode) nogil:
    """
    """
    pass

def hipStreamEndCapture_spt(stream,pGraph) nogil:
    """
    """
    pass

def hipStreamIsCapturing_spt(stream,pCaptureStatus) nogil:
    """
    """
    pass

def hipStreamGetCaptureInfo_spt(stream,pCaptureStatus,pId) nogil:
    """
    """
    pass

def hipStreamGetCaptureInfo_v2_spt(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out) nogil:
    """
    """
    pass

def hipLaunchHostFunc_spt(stream,fn,userData) nogil:
    """
    """
    pass

class hipDataType(enum.IntEnum):
    HIP_R_16F = 2
    HIP_R_32F = 0
    HIP_R_64F = 1
    HIP_C_16F = 6
    HIP_C_32F = 4
    HIP_C_64F = 5

class hipLibraryPropertyType(enum.IntEnum):
    HIP_LIBRARY_MAJOR_VERSION = 0
    HIP_LIBRARY_MINOR_VERSION = 1
    HIP_LIBRARY_PATCH_LEVEL = 2

def hipExtModuleLaunchKernel(f,globalWorkSizeX,globalWorkSizeY,globalWorkSizeZ,localWorkSizeX,localWorkSizeY,localWorkSizeZ,sharedMemBytes,hStream,kernelParams,extra,startEvent,stopEvent,flags) nogil:
    """@brief Launches kernel with parameters and shared memory on stream with arguments passed
    to kernel params or extra arguments.
    @param [in] f Kernel to launch.
    @param [in] gridDimX  X grid dimension specified in work-items.
    @param [in] gridDimY  Y grid dimension specified in work-items.
    @param [in] gridDimZ  Z grid dimension specified in work-items.
    @param [in] blockDimX  X block dimension specified in work-items.
    @param [in] blockDimY  Y grid dimension specified in work-items.
    @param [in] blockDimZ  Z grid dimension specified in work-items.
    @param [in] sharedMemBytes  Amount of dynamic shared memory to allocate for this kernel.
    HIP-Clang compiler provides support for extern shared declarations.
    @param [in] stream  Stream where the kernel should be dispatched.
    May be 0, in which case the default stream is used with associated synchronization rules.
    @param [in] kernelParams  pointer to kernel parameters.
    @param [in] extra  Pointer to kernel arguments. These are passed directly to the kernel and
    must be in the memory layout and alignment expected by the kernel.
    @param [in] startEvent  If non-null, specified event will be updated to track the start time of
    the kernel launch. The event must be created before calling this API.
    @param [in] stopEvent  If non-null, specified event will be updated to track the stop time of
    the kernel launch. The event must be created before calling this API.
    @param [in] flags. The value of hipExtAnyOrderLaunch, signifies if kernel can be
    launched in any order.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue.
    @warning kernellParams argument is not yet implemented in HIP, use extra instead.
    Please refer to hip_porting_driver_api.md for sample usage.
    HIP/ROCm actually updates the start event when the associated kernel completes.
    Currently, timing between startEvent and stopEvent does not include the time it takes to perform
    a system scope release/cache flush - only the time it takes to issues writes to cache.
    """
    pass

def hipHccModuleLaunchKernel(f,globalWorkSizeX,globalWorkSizeY,globalWorkSizeZ,localWorkSizeX,localWorkSizeY,localWorkSizeZ,sharedMemBytes,hStream,kernelParams,extra,startEvent,stopEvent) nogil:
    """@brief This HIP API is deprecated, please use hipExtModuleLaunchKernel() instead.
    """
    pass
