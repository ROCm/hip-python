# AMD_COPYRIGHT
from libc.stdint cimport *

cdef extern from "hip/hip_runtime_api.h":

    cdef int HIP_VERSION_MAJOR

    cdef int HIP_VERSION_MINOR

    cdef int HIP_VERSION_PATCH

    cdef int HIP_VERSION_BUILD_ID

    cdef int HIP_VERSION

    cdef int HIP_TRSA_OVERRIDE_FORMAT

    cdef int HIP_TRSF_READ_AS_INTEGER

    cdef int HIP_TRSF_NORMALIZED_COORDINATES

    cdef int HIP_TRSF_SRGB

    cdef int hipTextureType1D

    cdef int hipTextureType2D

    cdef int hipTextureType3D

    cdef int hipTextureTypeCubemap

    cdef int hipTextureType1DLayered

    cdef int hipTextureType2DLayered

    cdef int hipTextureTypeCubemapLayered

    cdef int HIP_IMAGE_OBJECT_SIZE_DWORD

    cdef int HIP_SAMPLER_OBJECT_SIZE_DWORD

    cdef int HIP_SAMPLER_OBJECT_OFFSET_DWORD

    cdef int HIP_TEXTURE_OBJECT_SIZE_DWORD

    cdef int hipIpcMemLazyEnablePeerAccess

    cdef int HIP_IPC_HANDLE_SIZE

    cdef int hipStreamDefault

    cdef int hipStreamNonBlocking

    cdef int hipEventDefault

    cdef int hipEventBlockingSync

    cdef int hipEventDisableTiming

    cdef int hipEventInterprocess

    cdef int hipEventReleaseToDevice

    cdef int hipEventReleaseToSystem

    cdef int hipHostMallocDefault

    cdef int hipHostMallocPortable

    cdef int hipHostMallocMapped

    cdef int hipHostMallocWriteCombined

    cdef int hipHostMallocNumaUser

    cdef int hipHostMallocCoherent

    cdef int hipHostMallocNonCoherent

    cdef int hipMemAttachGlobal

    cdef int hipMemAttachHost

    cdef int hipMemAttachSingle

    cdef int hipDeviceMallocDefault

    cdef int hipDeviceMallocFinegrained

    cdef int hipMallocSignalMemory

    cdef int hipHostRegisterDefault

    cdef int hipHostRegisterPortable

    cdef int hipHostRegisterMapped

    cdef int hipHostRegisterIoMemory

    cdef int hipExtHostRegisterCoarseGrained

    cdef int hipDeviceScheduleAuto

    cdef int hipDeviceScheduleSpin

    cdef int hipDeviceScheduleYield

    cdef int hipDeviceScheduleBlockingSync

    cdef int hipDeviceScheduleMask

    cdef int hipDeviceMapHost

    cdef int hipDeviceLmemResizeToMax

    cdef int hipArrayDefault

    cdef int hipArrayLayered

    cdef int hipArraySurfaceLoadStore

    cdef int hipArrayCubemap

    cdef int hipArrayTextureGather

    cdef int hipOccupancyDefault

    cdef int hipCooperativeLaunchMultiDeviceNoPreSync

    cdef int hipCooperativeLaunchMultiDeviceNoPostSync

    cdef int hipCpuDeviceId

    cdef int hipInvalidDeviceId

    cdef int hipExtAnyOrderLaunch

    cdef int hipStreamWaitValueGte

    cdef int hipStreamWaitValueEq

    cdef int hipStreamWaitValueAnd

    cdef int hipStreamWaitValueNor

cdef enum:
    HIP_SUCCESS
    HIP_ERROR_INVALID_VALUE
    HIP_ERROR_NOT_INITIALIZED
    HIP_ERROR_LAUNCH_OUT_OF_RESOURCES

cdef extern from "hip/hip_runtime_api.h":

    ctypedef struct hipDeviceArch_t:
        pass

    cdef struct hipUUID_t:
        char[16] bytes

    ctypedef hipUUID_t hipUUID

    cdef struct hipDeviceProp_t:
        char[256] name
        int totalGlobalMem
        int sharedMemPerBlock
        int regsPerBlock
        int warpSize
        int maxThreadsPerBlock
        int[3] maxThreadsDim
        int[3] maxGridSize
        int clockRate
        int memoryClockRate
        int memoryBusWidth
        int totalConstMem
        int major
        int minor
        int multiProcessorCount
        int l2CacheSize
        int maxThreadsPerMultiProcessor
        int computeMode
        int clockInstructionRate
        hipDeviceArch_t arch
        int concurrentKernels
        int pciDomainID
        int pciBusID
        int pciDeviceID
        int maxSharedMemoryPerMultiProcessor
        int isMultiGpuBoard
        int canMapHostMemory
        int gcnArch
        char[256] gcnArchName
        int integrated
        int cooperativeLaunch
        int cooperativeMultiDeviceLaunch
        int maxTexture1DLinear
        int maxTexture1D
        int[2] maxTexture2D
        int[3] maxTexture3D
        unsigned int * hdpMemFlushCntl
        unsigned int * hdpRegFlushCntl
        int memPitch
        int textureAlignment
        int texturePitchAlignment
        int kernelExecTimeoutEnabled
        int ECCEnabled
        int tccDriver
        int cooperativeMultiDeviceUnmatchedFunc
        int cooperativeMultiDeviceUnmatchedGridDim
        int cooperativeMultiDeviceUnmatchedBlockDim
        int cooperativeMultiDeviceUnmatchedSharedMem
        int isLargeBar
        int asicRevision
        int managedMemory
        int directManagedMemAccessFromHost
        int concurrentManagedAccess
        int pageableMemoryAccess
        int pageableMemoryAccessUsesHostPageTables

    cdef enum hipMemoryType:
        hipMemoryTypeHost
        hipMemoryTypeDevice
        hipMemoryTypeArray
        hipMemoryTypeUnified
        hipMemoryTypeManaged

    cdef struct hipPointerAttribute_t:
        hipMemoryType memoryType
        int device
        void * devicePointer
        void * hostPointer
        int isManaged
        unsigned int allocationFlags

    cdef enum hipError_t:
        hipSuccess
        hipErrorInvalidValue
        hipErrorOutOfMemory
        hipErrorMemoryAllocation
        hipErrorNotInitialized
        hipErrorInitializationError
        hipErrorDeinitialized
        hipErrorProfilerDisabled
        hipErrorProfilerNotInitialized
        hipErrorProfilerAlreadyStarted
        hipErrorProfilerAlreadyStopped
        hipErrorInvalidConfiguration
        hipErrorInvalidPitchValue
        hipErrorInvalidSymbol
        hipErrorInvalidDevicePointer
        hipErrorInvalidMemcpyDirection
        hipErrorInsufficientDriver
        hipErrorMissingConfiguration
        hipErrorPriorLaunchFailure
        hipErrorInvalidDeviceFunction
        hipErrorNoDevice
        hipErrorInvalidDevice
        hipErrorInvalidImage
        hipErrorInvalidContext
        hipErrorContextAlreadyCurrent
        hipErrorMapFailed
        hipErrorMapBufferObjectFailed
        hipErrorUnmapFailed
        hipErrorArrayIsMapped
        hipErrorAlreadyMapped
        hipErrorNoBinaryForGpu
        hipErrorAlreadyAcquired
        hipErrorNotMapped
        hipErrorNotMappedAsArray
        hipErrorNotMappedAsPointer
        hipErrorECCNotCorrectable
        hipErrorUnsupportedLimit
        hipErrorContextAlreadyInUse
        hipErrorPeerAccessUnsupported
        hipErrorInvalidKernelFile
        hipErrorInvalidGraphicsContext
        hipErrorInvalidSource
        hipErrorFileNotFound
        hipErrorSharedObjectSymbolNotFound
        hipErrorSharedObjectInitFailed
        hipErrorOperatingSystem
        hipErrorInvalidHandle
        hipErrorInvalidResourceHandle
        hipErrorIllegalState
        hipErrorNotFound
        hipErrorNotReady
        hipErrorIllegalAddress
        hipErrorLaunchOutOfResources
        hipErrorLaunchTimeOut
        hipErrorPeerAccessAlreadyEnabled
        hipErrorPeerAccessNotEnabled
        hipErrorSetOnActiveProcess
        hipErrorContextIsDestroyed
        hipErrorAssert
        hipErrorHostMemoryAlreadyRegistered
        hipErrorHostMemoryNotRegistered
        hipErrorLaunchFailure
        hipErrorCooperativeLaunchTooLarge
        hipErrorNotSupported
        hipErrorStreamCaptureUnsupported
        hipErrorStreamCaptureInvalidated
        hipErrorStreamCaptureMerge
        hipErrorStreamCaptureUnmatched
        hipErrorStreamCaptureUnjoined
        hipErrorStreamCaptureIsolation
        hipErrorStreamCaptureImplicit
        hipErrorCapturedEvent
        hipErrorStreamCaptureWrongThread
        hipErrorGraphExecUpdateFailure
        hipErrorUnknown
        hipErrorRuntimeMemory
        hipErrorRuntimeOther
        hipErrorTbd

    cdef enum hipDeviceAttribute_t:
        hipDeviceAttributeCudaCompatibleBegin
        hipDeviceAttributeEccEnabled
        hipDeviceAttributeAccessPolicyMaxWindowSize
        hipDeviceAttributeAsyncEngineCount
        hipDeviceAttributeCanMapHostMemory
        hipDeviceAttributeCanUseHostPointerForRegisteredMem
        hipDeviceAttributeClockRate
        hipDeviceAttributeComputeMode
        hipDeviceAttributeComputePreemptionSupported
        hipDeviceAttributeConcurrentKernels
        hipDeviceAttributeConcurrentManagedAccess
        hipDeviceAttributeCooperativeLaunch
        hipDeviceAttributeCooperativeMultiDeviceLaunch
        hipDeviceAttributeDeviceOverlap
        hipDeviceAttributeDirectManagedMemAccessFromHost
        hipDeviceAttributeGlobalL1CacheSupported
        hipDeviceAttributeHostNativeAtomicSupported
        hipDeviceAttributeIntegrated
        hipDeviceAttributeIsMultiGpuBoard
        hipDeviceAttributeKernelExecTimeout
        hipDeviceAttributeL2CacheSize
        hipDeviceAttributeLocalL1CacheSupported
        hipDeviceAttributeLuid
        hipDeviceAttributeLuidDeviceNodeMask
        hipDeviceAttributeComputeCapabilityMajor
        hipDeviceAttributeManagedMemory
        hipDeviceAttributeMaxBlocksPerMultiProcessor
        hipDeviceAttributeMaxBlockDimX
        hipDeviceAttributeMaxBlockDimY
        hipDeviceAttributeMaxBlockDimZ
        hipDeviceAttributeMaxGridDimX
        hipDeviceAttributeMaxGridDimY
        hipDeviceAttributeMaxGridDimZ
        hipDeviceAttributeMaxSurface1D
        hipDeviceAttributeMaxSurface1DLayered
        hipDeviceAttributeMaxSurface2D
        hipDeviceAttributeMaxSurface2DLayered
        hipDeviceAttributeMaxSurface3D
        hipDeviceAttributeMaxSurfaceCubemap
        hipDeviceAttributeMaxSurfaceCubemapLayered
        hipDeviceAttributeMaxTexture1DWidth
        hipDeviceAttributeMaxTexture1DLayered
        hipDeviceAttributeMaxTexture1DLinear
        hipDeviceAttributeMaxTexture1DMipmap
        hipDeviceAttributeMaxTexture2DWidth
        hipDeviceAttributeMaxTexture2DHeight
        hipDeviceAttributeMaxTexture2DGather
        hipDeviceAttributeMaxTexture2DLayered
        hipDeviceAttributeMaxTexture2DLinear
        hipDeviceAttributeMaxTexture2DMipmap
        hipDeviceAttributeMaxTexture3DWidth
        hipDeviceAttributeMaxTexture3DHeight
        hipDeviceAttributeMaxTexture3DDepth
        hipDeviceAttributeMaxTexture3DAlt
        hipDeviceAttributeMaxTextureCubemap
        hipDeviceAttributeMaxTextureCubemapLayered
        hipDeviceAttributeMaxThreadsDim
        hipDeviceAttributeMaxThreadsPerBlock
        hipDeviceAttributeMaxThreadsPerMultiProcessor
        hipDeviceAttributeMaxPitch
        hipDeviceAttributeMemoryBusWidth
        hipDeviceAttributeMemoryClockRate
        hipDeviceAttributeComputeCapabilityMinor
        hipDeviceAttributeMultiGpuBoardGroupID
        hipDeviceAttributeMultiprocessorCount
        hipDeviceAttributeName
        hipDeviceAttributePageableMemoryAccess
        hipDeviceAttributePageableMemoryAccessUsesHostPageTables
        hipDeviceAttributePciBusId
        hipDeviceAttributePciDeviceId
        hipDeviceAttributePciDomainID
        hipDeviceAttributePersistingL2CacheMaxSize
        hipDeviceAttributeMaxRegistersPerBlock
        hipDeviceAttributeMaxRegistersPerMultiprocessor
        hipDeviceAttributeReservedSharedMemPerBlock
        hipDeviceAttributeMaxSharedMemoryPerBlock
        hipDeviceAttributeSharedMemPerBlockOptin
        hipDeviceAttributeSharedMemPerMultiprocessor
        hipDeviceAttributeSingleToDoublePrecisionPerfRatio
        hipDeviceAttributeStreamPrioritiesSupported
        hipDeviceAttributeSurfaceAlignment
        hipDeviceAttributeTccDriver
        hipDeviceAttributeTextureAlignment
        hipDeviceAttributeTexturePitchAlignment
        hipDeviceAttributeTotalConstantMemory
        hipDeviceAttributeTotalGlobalMem
        hipDeviceAttributeUnifiedAddressing
        hipDeviceAttributeUuid
        hipDeviceAttributeWarpSize
        hipDeviceAttributeMemoryPoolsSupported
        hipDeviceAttributeVirtualMemoryManagementSupported
        hipDeviceAttributeCudaCompatibleEnd
        hipDeviceAttributeAmdSpecificBegin
        hipDeviceAttributeClockInstructionRate
        hipDeviceAttributeArch
        hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
        hipDeviceAttributeGcnArch
        hipDeviceAttributeGcnArchName
        hipDeviceAttributeHdpMemFlushCntl
        hipDeviceAttributeHdpRegFlushCntl
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem
        hipDeviceAttributeIsLargeBar
        hipDeviceAttributeAsicRevision
        hipDeviceAttributeCanUseStreamWaitValue
        hipDeviceAttributeImageSupport
        hipDeviceAttributePhysicalMultiProcessorCount
        hipDeviceAttributeFineGrainSupport
        hipDeviceAttributeWallClockRate
        hipDeviceAttributeAmdSpecificEnd
        hipDeviceAttributeVendorSpecificBegin

    cdef enum hipComputeMode:
        hipComputeModeDefault
        hipComputeModeExclusive
        hipComputeModeProhibited
        hipComputeModeExclusiveProcess

    ctypedef void * hipDeviceptr_t

    cdef enum hipChannelFormatKind:
        hipChannelFormatKindSigned
        hipChannelFormatKindUnsigned
        hipChannelFormatKindFloat
        hipChannelFormatKindNone

    cdef struct hipChannelFormatDesc:
        int x
        int y
        int z
        int w
        hipChannelFormatKind f

    cdef enum hipArray_Format:
        HIP_AD_FORMAT_UNSIGNED_INT8
        HIP_AD_FORMAT_UNSIGNED_INT16
        HIP_AD_FORMAT_UNSIGNED_INT32
        HIP_AD_FORMAT_SIGNED_INT8
        HIP_AD_FORMAT_SIGNED_INT16
        HIP_AD_FORMAT_SIGNED_INT32
        HIP_AD_FORMAT_HALF
        HIP_AD_FORMAT_FLOAT

    cdef struct HIP_ARRAY_DESCRIPTOR:
        int Width
        int Height
        hipArray_Format Format
        unsigned int NumChannels

    cdef struct HIP_ARRAY3D_DESCRIPTOR:
        int Width
        int Height
        int Depth
        hipArray_Format Format
        unsigned int NumChannels
        unsigned int Flags

    cdef struct hipArray:
        void * data
        hipChannelFormatDesc desc
        unsigned int type
        unsigned int width
        unsigned int height
        unsigned int depth
        hipArray_Format Format
        unsigned int NumChannels
        int isDrv
        unsigned int textureType

    cdef struct hip_Memcpy2D:
        int srcXInBytes
        int srcY
        hipMemoryType srcMemoryType
        const void * srcHost
        hipDeviceptr_t srcDevice
        hipArray * srcArray
        int srcPitch
        int dstXInBytes
        int dstY
        hipMemoryType dstMemoryType
        void * dstHost
        hipDeviceptr_t dstDevice
        hipArray * dstArray
        int dstPitch
        int WidthInBytes
        int Height

    ctypedef hipArray * hipArray_t

    ctypedef hipArray_t hiparray

    ctypedef hipArray * hipArray_const_t

    cdef struct hipMipmappedArray:
        void * data
        hipChannelFormatDesc desc
        unsigned int type
        unsigned int width
        unsigned int height
        unsigned int depth
        unsigned int min_mipmap_level
        unsigned int max_mipmap_level
        unsigned int flags
        hipArray_Format format

    ctypedef hipMipmappedArray * hipMipmappedArray_t

    ctypedef hipMipmappedArray * hipMipmappedArray_const_t

    cdef enum hipResourceType:
        hipResourceTypeArray
        hipResourceTypeMipmappedArray
        hipResourceTypeLinear
        hipResourceTypePitch2D

    cdef enum HIPresourcetype_enum:
        HIP_RESOURCE_TYPE_ARRAY
        HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
        HIP_RESOURCE_TYPE_LINEAR
        HIP_RESOURCE_TYPE_PITCH2D

    ctypedef HIPresourcetype_enum HIPresourcetype

    ctypedef HIPresourcetype_enum hipResourcetype

    cdef enum HIPaddress_mode_enum:
        HIP_TR_ADDRESS_MODE_WRAP
        HIP_TR_ADDRESS_MODE_CLAMP
        HIP_TR_ADDRESS_MODE_MIRROR
        HIP_TR_ADDRESS_MODE_BORDER

    ctypedef HIPaddress_mode_enum HIPaddress_mode

    cdef enum HIPfilter_mode_enum:
        HIP_TR_FILTER_MODE_POINT
        HIP_TR_FILTER_MODE_LINEAR

    ctypedef HIPfilter_mode_enum HIPfilter_mode

    cdef struct HIP_TEXTURE_DESC_st:
        HIPaddress_mode_enum[3] addressMode
        HIPfilter_mode_enum filterMode
        unsigned int flags
        unsigned int maxAnisotropy
        HIPfilter_mode_enum mipmapFilterMode
        float mipmapLevelBias
        float minMipmapLevelClamp
        float maxMipmapLevelClamp
        float[4] borderColor
        int[12] reserved

    ctypedef HIP_TEXTURE_DESC_st HIP_TEXTURE_DESC

    cdef enum hipResourceViewFormat:
        hipResViewFormatNone
        hipResViewFormatUnsignedChar1
        hipResViewFormatUnsignedChar2
        hipResViewFormatUnsignedChar4
        hipResViewFormatSignedChar1
        hipResViewFormatSignedChar2
        hipResViewFormatSignedChar4
        hipResViewFormatUnsignedShort1
        hipResViewFormatUnsignedShort2
        hipResViewFormatUnsignedShort4
        hipResViewFormatSignedShort1
        hipResViewFormatSignedShort2
        hipResViewFormatSignedShort4
        hipResViewFormatUnsignedInt1
        hipResViewFormatUnsignedInt2
        hipResViewFormatUnsignedInt4
        hipResViewFormatSignedInt1
        hipResViewFormatSignedInt2
        hipResViewFormatSignedInt4
        hipResViewFormatHalf1
        hipResViewFormatHalf2
        hipResViewFormatHalf4
        hipResViewFormatFloat1
        hipResViewFormatFloat2
        hipResViewFormatFloat4
        hipResViewFormatUnsignedBlockCompressed1
        hipResViewFormatUnsignedBlockCompressed2
        hipResViewFormatUnsignedBlockCompressed3
        hipResViewFormatUnsignedBlockCompressed4
        hipResViewFormatSignedBlockCompressed4
        hipResViewFormatUnsignedBlockCompressed5
        hipResViewFormatSignedBlockCompressed5
        hipResViewFormatUnsignedBlockCompressed6H
        hipResViewFormatSignedBlockCompressed6H
        hipResViewFormatUnsignedBlockCompressed7

    cdef enum HIPresourceViewFormat_enum:
        HIP_RES_VIEW_FORMAT_NONE
        HIP_RES_VIEW_FORMAT_UINT_1X8
        HIP_RES_VIEW_FORMAT_UINT_2X8
        HIP_RES_VIEW_FORMAT_UINT_4X8
        HIP_RES_VIEW_FORMAT_SINT_1X8
        HIP_RES_VIEW_FORMAT_SINT_2X8
        HIP_RES_VIEW_FORMAT_SINT_4X8
        HIP_RES_VIEW_FORMAT_UINT_1X16
        HIP_RES_VIEW_FORMAT_UINT_2X16
        HIP_RES_VIEW_FORMAT_UINT_4X16
        HIP_RES_VIEW_FORMAT_SINT_1X16
        HIP_RES_VIEW_FORMAT_SINT_2X16
        HIP_RES_VIEW_FORMAT_SINT_4X16
        HIP_RES_VIEW_FORMAT_UINT_1X32
        HIP_RES_VIEW_FORMAT_UINT_2X32
        HIP_RES_VIEW_FORMAT_UINT_4X32
        HIP_RES_VIEW_FORMAT_SINT_1X32
        HIP_RES_VIEW_FORMAT_SINT_2X32
        HIP_RES_VIEW_FORMAT_SINT_4X32
        HIP_RES_VIEW_FORMAT_FLOAT_1X16
        HIP_RES_VIEW_FORMAT_FLOAT_2X16
        HIP_RES_VIEW_FORMAT_FLOAT_4X16
        HIP_RES_VIEW_FORMAT_FLOAT_1X32
        HIP_RES_VIEW_FORMAT_FLOAT_2X32
        HIP_RES_VIEW_FORMAT_FLOAT_4X32
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
        HIP_RES_VIEW_FORMAT_SIGNED_BC4
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
        HIP_RES_VIEW_FORMAT_SIGNED_BC5
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
        HIP_RES_VIEW_FORMAT_SIGNED_BC6H
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC7

    ctypedef HIPresourceViewFormat_enum HIPresourceViewFormat

cdef struct hipResourceDesc_union_0_struct_0:
    hipArray_t array

cdef struct hipResourceDesc_union_0_struct_1:
    hipMipmappedArray_t mipmap

cdef struct hipResourceDesc_union_0_struct_2:
    void * devPtr
    hipChannelFormatDesc desc
    int sizeInBytes

cdef struct hipResourceDesc_union_0_struct_3:
    void * devPtr
    hipChannelFormatDesc desc
    int width
    int height
    int pitchInBytes

cdef union hipResourceDesc_union_0:
    hipResourceDesc_union_0_struct_0 array
    hipResourceDesc_union_0_struct_1 mipmap
    hipResourceDesc_union_0_struct_2 linear
    hipResourceDesc_union_0_struct_3 pitch2D

cdef extern from "hip/hip_runtime_api.h":

    cdef struct hipResourceDesc:
        hipResourceType resType
        hipResourceDesc_union_0 res

cdef struct HIP_RESOURCE_DESC_st_union_0_struct_0:
    hipArray_t hArray

cdef struct HIP_RESOURCE_DESC_st_union_0_struct_1:
    hipMipmappedArray_t hMipmappedArray

cdef struct HIP_RESOURCE_DESC_st_union_0_struct_2:
    hipDeviceptr_t devPtr
    hipArray_Format format
    unsigned int numChannels
    int sizeInBytes

cdef struct HIP_RESOURCE_DESC_st_union_0_struct_3:
    hipDeviceptr_t devPtr
    hipArray_Format format
    unsigned int numChannels
    int width
    int height
    int pitchInBytes

cdef struct HIP_RESOURCE_DESC_st_union_0_struct_4:
    int[32] reserved

cdef union HIP_RESOURCE_DESC_st_union_0:
    HIP_RESOURCE_DESC_st_union_0_struct_0 array
    HIP_RESOURCE_DESC_st_union_0_struct_1 mipmap
    HIP_RESOURCE_DESC_st_union_0_struct_2 linear
    HIP_RESOURCE_DESC_st_union_0_struct_3 pitch2D
    HIP_RESOURCE_DESC_st_union_0_struct_4 reserved

cdef extern from "hip/hip_runtime_api.h":

    cdef struct HIP_RESOURCE_DESC_st:
        HIPresourcetype_enum resType
        HIP_RESOURCE_DESC_st_union_0 res
        unsigned int flags

    ctypedef HIP_RESOURCE_DESC_st HIP_RESOURCE_DESC

    cdef struct hipResourceViewDesc:
        hipResourceViewFormat format
        int width
        int height
        int depth
        unsigned int firstMipmapLevel
        unsigned int lastMipmapLevel
        unsigned int firstLayer
        unsigned int lastLayer

    cdef struct HIP_RESOURCE_VIEW_DESC_st:
        HIPresourceViewFormat_enum format
        int width
        int height
        int depth
        unsigned int firstMipmapLevel
        unsigned int lastMipmapLevel
        unsigned int firstLayer
        unsigned int lastLayer
        unsigned int[16] reserved

    ctypedef HIP_RESOURCE_VIEW_DESC_st HIP_RESOURCE_VIEW_DESC

    cdef enum hipMemcpyKind:
        hipMemcpyHostToHost
        hipMemcpyHostToDevice
        hipMemcpyDeviceToHost
        hipMemcpyDeviceToDevice
        hipMemcpyDefault

    cdef struct hipPitchedPtr:
        void * ptr
        int pitch
        int xsize
        int ysize

    cdef struct hipExtent:
        int width
        int height
        int depth

    cdef struct hipPos:
        int x
        int y
        int z

    cdef struct hipMemcpy3DParms:
        hipArray_t srcArray
        hipPos srcPos
        hipPitchedPtr srcPtr
        hipArray_t dstArray
        hipPos dstPos
        hipPitchedPtr dstPtr
        hipExtent extent
        hipMemcpyKind kind

    cdef struct HIP_MEMCPY3D:
        unsigned int srcXInBytes
        unsigned int srcY
        unsigned int srcZ
        unsigned int srcLOD
        hipMemoryType srcMemoryType
        const void * srcHost
        hipDeviceptr_t srcDevice
        hipArray_t srcArray
        unsigned int srcPitch
        unsigned int srcHeight
        unsigned int dstXInBytes
        unsigned int dstY
        unsigned int dstZ
        unsigned int dstLOD
        hipMemoryType dstMemoryType
        void * dstHost
        hipDeviceptr_t dstDevice
        hipArray_t dstArray
        unsigned int dstPitch
        unsigned int dstHeight
        unsigned int WidthInBytes
        unsigned int Height
        unsigned int Depth

    cdef enum hipFunction_attribute:
        HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_NUM_REGS
        HIP_FUNC_ATTRIBUTE_PTX_VERSION
        HIP_FUNC_ATTRIBUTE_BINARY_VERSION
        HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
        HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        HIP_FUNC_ATTRIBUTE_MAX

    cdef enum hipPointer_attribute:
        HIP_POINTER_ATTRIBUTE_CONTEXT
        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
        HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
        HIP_POINTER_ATTRIBUTE_HOST_POINTER
        HIP_POINTER_ATTRIBUTE_P2P_TOKENS
        HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
        HIP_POINTER_ATTRIBUTE_BUFFER_ID
        HIP_POINTER_ATTRIBUTE_IS_MANAGED
        HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
        HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
        HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
        HIP_POINTER_ATTRIBUTE_RANGE_SIZE
        HIP_POINTER_ATTRIBUTE_MAPPED
        HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
        HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
        HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
        HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE

    ctypedef struct uchar1:
        pass

    ctypedef struct uchar2:
        pass

    ctypedef struct uchar3:
        pass

    ctypedef struct uchar4:
        pass

    ctypedef struct char1:
        pass

    ctypedef struct char2:
        pass

    ctypedef struct char3:
        pass

    ctypedef struct char4:
        pass

    ctypedef struct ushort1:
        pass

    ctypedef struct ushort2:
        pass

    ctypedef struct ushort3:
        pass

    ctypedef struct ushort4:
        pass

    ctypedef struct short1:
        pass

    ctypedef struct short2:
        pass

    ctypedef struct short3:
        pass

    ctypedef struct short4:
        pass

    ctypedef struct uint1:
        pass

    ctypedef struct uint2:
        pass

    ctypedef struct uint3:
        pass

    ctypedef struct uint4:
        pass

    ctypedef struct int1:
        pass

    ctypedef struct int2:
        pass

    ctypedef struct int3:
        pass

    ctypedef struct int4:
        pass

    ctypedef struct ulong1:
        pass

    ctypedef struct ulong2:
        pass

    ctypedef struct ulong3:
        pass

    ctypedef struct ulong4:
        pass

    ctypedef struct long1:
        pass

    ctypedef struct long2:
        pass

    ctypedef struct long3:
        pass

    ctypedef struct long4:
        pass

    ctypedef struct ulonglong1:
        pass

    ctypedef struct ulonglong2:
        pass

    ctypedef struct ulonglong3:
        pass

    ctypedef struct ulonglong4:
        pass

    ctypedef struct longlong1:
        pass

    ctypedef struct longlong2:
        pass

    ctypedef struct longlong3:
        pass

    ctypedef struct longlong4:
        pass

    ctypedef struct float1:
        pass

    ctypedef struct float2:
        pass

    ctypedef struct float3:
        pass

    ctypedef struct float4:
        pass

    ctypedef struct double1:
        pass

    ctypedef struct double2:
        pass

    ctypedef struct double3:
        pass

    ctypedef struct double4:
        pass


    hipChannelFormatDesc hipCreateChannelDesc(int x,int y,int z,int w,hipChannelFormatKind f) nogil


    cdef struct __hip_texture:
        pass

    ctypedef __hip_texture * hipTextureObject_t

    cdef enum hipTextureAddressMode:
        hipAddressModeWrap
        hipAddressModeClamp
        hipAddressModeMirror
        hipAddressModeBorder

    cdef enum hipTextureFilterMode:
        hipFilterModePoint
        hipFilterModeLinear

    cdef enum hipTextureReadMode:
        hipReadModeElementType
        hipReadModeNormalizedFloat

    cdef struct textureReference:
        int normalized
        hipTextureReadMode readMode
        hipTextureFilterMode filterMode
        hipTextureAddressMode[3] addressMode
        hipChannelFormatDesc channelDesc
        int sRGB
        unsigned int maxAnisotropy
        hipTextureFilterMode mipmapFilterMode
        float mipmapLevelBias
        float minMipmapLevelClamp
        float maxMipmapLevelClamp
        hipTextureObject_t textureObject
        int numChannels
        hipArray_Format format

    cdef struct hipTextureDesc:
        hipTextureAddressMode[3] addressMode
        hipTextureFilterMode filterMode
        hipTextureReadMode readMode
        int sRGB
        float[4] borderColor
        int normalizedCoords
        unsigned int maxAnisotropy
        hipTextureFilterMode mipmapFilterMode
        float mipmapLevelBias
        float minMipmapLevelClamp
        float maxMipmapLevelClamp

    cdef struct __hip_surface:
        pass

    ctypedef __hip_surface * hipSurfaceObject_t

    cdef struct surfaceReference:
        hipSurfaceObject_t surfaceObject

    cdef enum hipSurfaceBoundaryMode:
        hipBoundaryModeZero
        hipBoundaryModeTrap
        hipBoundaryModeClamp

    cdef struct ihipCtx_t:
        pass

    ctypedef ihipCtx_t * hipCtx_t

    ctypedef int hipDevice_t

    cdef enum hipDeviceP2PAttr:
        hipDevP2PAttrPerformanceRank
        hipDevP2PAttrAccessSupported
        hipDevP2PAttrNativeAtomicSupported
        hipDevP2PAttrHipArrayAccessSupported

    cdef struct ihipStream_t:
        pass

    ctypedef ihipStream_t * hipStream_t

    cdef struct hipIpcMemHandle_st:
        char[64] reserved

    ctypedef hipIpcMemHandle_st hipIpcMemHandle_t

    cdef struct hipIpcEventHandle_st:
        char[64] reserved

    ctypedef hipIpcEventHandle_st hipIpcEventHandle_t

    cdef struct ihipModule_t:
        pass

    ctypedef ihipModule_t * hipModule_t

    cdef struct ihipModuleSymbol_t:
        pass

    ctypedef ihipModuleSymbol_t * hipFunction_t

    cdef struct ihipMemPoolHandle_t:
        pass

    ctypedef ihipMemPoolHandle_t * hipMemPool_t

    cdef struct hipFuncAttributes:
        int binaryVersion
        int cacheModeCA
        int constSizeBytes
        int localSizeBytes
        int maxDynamicSharedSizeBytes
        int maxThreadsPerBlock
        int numRegs
        int preferredShmemCarveout
        int ptxVersion
        int sharedSizeBytes

    cdef struct ihipEvent_t:
        pass

    ctypedef ihipEvent_t * hipEvent_t

    cdef enum hipLimit_t:
        hipLimitStackSize
        hipLimitPrintfFifoSize
        hipLimitMallocHeapSize
        hipLimitRange

    cdef enum hipMemoryAdvise:
        hipMemAdviseSetReadMostly
        hipMemAdviseUnsetReadMostly
        hipMemAdviseSetPreferredLocation
        hipMemAdviseUnsetPreferredLocation
        hipMemAdviseSetAccessedBy
        hipMemAdviseUnsetAccessedBy
        hipMemAdviseSetCoarseGrain
        hipMemAdviseUnsetCoarseGrain

    cdef enum hipMemRangeCoherencyMode:
        hipMemRangeCoherencyModeFineGrain
        hipMemRangeCoherencyModeCoarseGrain
        hipMemRangeCoherencyModeIndeterminate

    cdef enum hipMemRangeAttribute:
        hipMemRangeAttributeReadMostly
        hipMemRangeAttributePreferredLocation
        hipMemRangeAttributeAccessedBy
        hipMemRangeAttributeLastPrefetchLocation
        hipMemRangeAttributeCoherencyMode

    cdef enum hipMemPoolAttr:
        hipMemPoolReuseFollowEventDependencies
        hipMemPoolReuseAllowOpportunistic
        hipMemPoolReuseAllowInternalDependencies
        hipMemPoolAttrReleaseThreshold
        hipMemPoolAttrReservedMemCurrent
        hipMemPoolAttrReservedMemHigh
        hipMemPoolAttrUsedMemCurrent
        hipMemPoolAttrUsedMemHigh

    cdef enum hipMemLocationType:
        hipMemLocationTypeInvalid
        hipMemLocationTypeDevice

    cdef struct hipMemLocation:
        hipMemLocationType type
        int id

    cdef enum hipMemAccessFlags:
        hipMemAccessFlagsProtNone
        hipMemAccessFlagsProtRead
        hipMemAccessFlagsProtReadWrite

    cdef struct hipMemAccessDesc:
        hipMemLocation location
        hipMemAccessFlags flags

    cdef enum hipMemAllocationType:
        hipMemAllocationTypeInvalid
        hipMemAllocationTypePinned
        hipMemAllocationTypeMax

    cdef enum hipMemAllocationHandleType:
        hipMemHandleTypeNone
        hipMemHandleTypePosixFileDescriptor
        hipMemHandleTypeWin32
        hipMemHandleTypeWin32Kmt

    cdef struct hipMemPoolProps:
        hipMemAllocationType allocType
        hipMemAllocationHandleType handleTypes
        hipMemLocation location
        void * win32SecurityAttributes
        unsigned char[64] reserved

    cdef struct hipMemPoolPtrExportData:
        unsigned char[64] reserved

    cdef enum hipJitOption:
        hipJitOptionMaxRegisters
        hipJitOptionThreadsPerBlock
        hipJitOptionWallTime
        hipJitOptionInfoLogBuffer
        hipJitOptionInfoLogBufferSizeBytes
        hipJitOptionErrorLogBuffer
        hipJitOptionErrorLogBufferSizeBytes
        hipJitOptionOptimizationLevel
        hipJitOptionTargetFromContext
        hipJitOptionTarget
        hipJitOptionFallbackStrategy
        hipJitOptionGenerateDebugInfo
        hipJitOptionLogVerbose
        hipJitOptionGenerateLineInfo
        hipJitOptionCacheMode
        hipJitOptionSm3xOpt
        hipJitOptionFastCompile
        hipJitOptionNumOptions

    cdef enum hipFuncAttribute:
        hipFuncAttributeMaxDynamicSharedMemorySize
        hipFuncAttributePreferredSharedMemoryCarveout
        hipFuncAttributeMax

    cdef enum hipFuncCache_t:
        hipFuncCachePreferNone
        hipFuncCachePreferShared
        hipFuncCachePreferL1
        hipFuncCachePreferEqual

    cdef enum hipSharedMemConfig:
        hipSharedMemBankSizeDefault
        hipSharedMemBankSizeFourByte
        hipSharedMemBankSizeEightByte

    cdef struct dim3:
        uint32_t x
        uint32_t y
        uint32_t z

    cdef struct hipLaunchParams_t:
        void * func
        dim3 gridDim
        dim3 blockDim
        void ** args
        int sharedMem
        hipStream_t stream

    ctypedef hipLaunchParams_t hipLaunchParams

    cdef enum hipExternalMemoryHandleType_enum:
        hipExternalMemoryHandleTypeOpaqueFd
        hipExternalMemoryHandleTypeOpaqueWin32
        hipExternalMemoryHandleTypeOpaqueWin32Kmt
        hipExternalMemoryHandleTypeD3D12Heap
        hipExternalMemoryHandleTypeD3D12Resource
        hipExternalMemoryHandleTypeD3D11Resource
        hipExternalMemoryHandleTypeD3D11ResourceKmt

    ctypedef hipExternalMemoryHandleType_enum hipExternalMemoryHandleType

cdef struct hipExternalMemoryHandleDesc_st_union_0_struct_0:
    void * handle
    const void * name

cdef union hipExternalMemoryHandleDesc_st_union_0:
    int fd
    hipExternalMemoryHandleDesc_st_union_0_struct_0 win32

cdef extern from "hip/hip_runtime_api.h":

    cdef struct hipExternalMemoryHandleDesc_st:
        hipExternalMemoryHandleType_enum type
        hipExternalMemoryHandleDesc_st_union_0 handle
        unsigned long long size
        unsigned int flags

    ctypedef hipExternalMemoryHandleDesc_st hipExternalMemoryHandleDesc

    cdef struct hipExternalMemoryBufferDesc_st:
        unsigned long long offset
        unsigned long long size
        unsigned int flags

    ctypedef hipExternalMemoryBufferDesc_st hipExternalMemoryBufferDesc

    ctypedef void * hipExternalMemory_t

    cdef enum hipExternalSemaphoreHandleType_enum:
        hipExternalSemaphoreHandleTypeOpaqueFd
        hipExternalSemaphoreHandleTypeOpaqueWin32
        hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
        hipExternalSemaphoreHandleTypeD3D12Fence

    ctypedef hipExternalSemaphoreHandleType_enum hipExternalSemaphoreHandleType

cdef struct hipExternalSemaphoreHandleDesc_st_union_0_struct_0:
    void * handle
    const void * name

cdef union hipExternalSemaphoreHandleDesc_st_union_0:
    int fd
    hipExternalSemaphoreHandleDesc_st_union_0_struct_0 win32

cdef extern from "hip/hip_runtime_api.h":

    cdef struct hipExternalSemaphoreHandleDesc_st:
        hipExternalSemaphoreHandleType_enum type
        hipExternalSemaphoreHandleDesc_st_union_0 handle
        unsigned int flags

    ctypedef hipExternalSemaphoreHandleDesc_st hipExternalSemaphoreHandleDesc

    ctypedef void * hipExternalSemaphore_t

cdef struct hipExternalSemaphoreSignalParams_st_struct_0_struct_0:
    unsigned long long value

cdef struct hipExternalSemaphoreSignalParams_st_struct_0_struct_1:
    unsigned long long key

cdef struct hipExternalSemaphoreSignalParams_st_struct_0:
    hipExternalSemaphoreSignalParams_st_struct_0_struct_0 fence
    hipExternalSemaphoreSignalParams_st_struct_0_struct_1 keyedMutex
    unsigned int[12] reserved

cdef extern from "hip/hip_runtime_api.h":

    cdef struct hipExternalSemaphoreSignalParams_st:
        hipExternalSemaphoreSignalParams_st_struct_0 params
        unsigned int flags
        unsigned int[16] reserved

    ctypedef hipExternalSemaphoreSignalParams_st hipExternalSemaphoreSignalParams

cdef struct hipExternalSemaphoreWaitParams_st_struct_0_struct_0:
    unsigned long long value

cdef struct hipExternalSemaphoreWaitParams_st_struct_0_struct_1:
    unsigned long long key
    unsigned int timeoutMs

cdef struct hipExternalSemaphoreWaitParams_st_struct_0:
    hipExternalSemaphoreWaitParams_st_struct_0_struct_0 fence
    hipExternalSemaphoreWaitParams_st_struct_0_struct_1 keyedMutex
    unsigned int[10] reserved

cdef extern from "hip/hip_runtime_api.h":

    cdef struct hipExternalSemaphoreWaitParams_st:
        hipExternalSemaphoreWaitParams_st_struct_0 params
        unsigned int flags
        unsigned int[16] reserved

    ctypedef hipExternalSemaphoreWaitParams_st hipExternalSemaphoreWaitParams

    cdef enum hipGLDeviceList:
        hipGLDeviceListAll
        hipGLDeviceListCurrentFrame
        hipGLDeviceListNextFrame

    cdef enum hipGraphicsRegisterFlags:
        hipGraphicsRegisterFlagsNone
        hipGraphicsRegisterFlagsReadOnly
        hipGraphicsRegisterFlagsWriteDiscard
        hipGraphicsRegisterFlagsSurfaceLoadStore
        hipGraphicsRegisterFlagsTextureGather

    cdef struct _hipGraphicsResource:
        pass

    ctypedef _hipGraphicsResource hipGraphicsResource

    ctypedef _hipGraphicsResource * hipGraphicsResource_t

    cdef struct ihipGraph:
        pass

    ctypedef ihipGraph * hipGraph_t

    cdef struct hipGraphNode:
        pass

    ctypedef hipGraphNode * hipGraphNode_t

    cdef struct hipGraphExec:
        pass

    ctypedef hipGraphExec * hipGraphExec_t

    cdef struct hipUserObject:
        pass

    ctypedef hipUserObject * hipUserObject_t

    cdef enum hipGraphNodeType:
        hipGraphNodeTypeKernel
        hipGraphNodeTypeMemcpy
        hipGraphNodeTypeMemset
        hipGraphNodeTypeHost
        hipGraphNodeTypeGraph
        hipGraphNodeTypeEmpty
        hipGraphNodeTypeWaitEvent
        hipGraphNodeTypeEventRecord
        hipGraphNodeTypeExtSemaphoreSignal
        hipGraphNodeTypeExtSemaphoreWait
        hipGraphNodeTypeMemcpyFromSymbol
        hipGraphNodeTypeMemcpyToSymbol
        hipGraphNodeTypeCount

    ctypedef void (*hipHostFn_t) (void *)

    cdef struct hipHostNodeParams:
        hipHostFn_t fn
        void * userData

    cdef struct hipKernelNodeParams:
        dim3 blockDim
        void ** extra
        void * func
        dim3 gridDim
        void ** kernelParams
        unsigned int sharedMemBytes

    cdef struct hipMemsetParams:
        void * dst
        unsigned int elementSize
        int height
        int pitch
        unsigned int value
        int width

    cdef enum hipKernelNodeAttrID:
        hipKernelNodeAttributeAccessPolicyWindow
        hipKernelNodeAttributeCooperative

    cdef enum hipAccessProperty:
        hipAccessPropertyNormal
        hipAccessPropertyStreaming
        hipAccessPropertyPersisting

    cdef struct hipAccessPolicyWindow:
        void * base_ptr
        hipAccessProperty hitProp
        float hitRatio
        hipAccessProperty missProp
        int num_bytes

    cdef union hipKernelNodeAttrValue:
        hipAccessPolicyWindow accessPolicyWindow
        int cooperative

    cdef enum hipGraphExecUpdateResult:
        hipGraphExecUpdateSuccess
        hipGraphExecUpdateError
        hipGraphExecUpdateErrorTopologyChanged
        hipGraphExecUpdateErrorNodeTypeChanged
        hipGraphExecUpdateErrorFunctionChanged
        hipGraphExecUpdateErrorParametersChanged
        hipGraphExecUpdateErrorNotSupported
        hipGraphExecUpdateErrorUnsupportedFunctionChange

    cdef enum hipStreamCaptureMode:
        hipStreamCaptureModeGlobal
        hipStreamCaptureModeThreadLocal
        hipStreamCaptureModeRelaxed

    cdef enum hipStreamCaptureStatus:
        hipStreamCaptureStatusNone
        hipStreamCaptureStatusActive
        hipStreamCaptureStatusInvalidated

    cdef enum hipStreamUpdateCaptureDependenciesFlags:
        hipStreamAddCaptureDependencies
        hipStreamSetCaptureDependencies

    cdef enum hipGraphMemAttributeType:
        hipGraphMemAttrUsedMemCurrent
        hipGraphMemAttrUsedMemHigh
        hipGraphMemAttrReservedMemCurrent
        hipGraphMemAttrReservedMemHigh

    cdef enum hipUserObjectFlags:
        hipUserObjectNoDestructorSync

    cdef enum hipUserObjectRetainFlags:
        hipGraphUserObjectMove

    cdef enum hipGraphInstantiateFlags:
        hipGraphInstantiateFlagAutoFreeOnLaunch

cdef struct hipMemAllocationProp_struct_0:
    unsigned char compressionType
    unsigned char gpuDirectRDMACapable
    unsigned short usage

cdef extern from "hip/hip_runtime_api.h":

    cdef struct hipMemAllocationProp:
        hipMemAllocationType type
        hipMemAllocationHandleType requestedHandleType
        hipMemLocation location
        void * win32HandleMetaData
        hipMemAllocationProp_struct_0 allocFlags

    cdef struct ihipMemGenericAllocationHandle:
        pass

    ctypedef ihipMemGenericAllocationHandle * hipMemGenericAllocationHandle_t

    cdef enum hipMemAllocationGranularity_flags:
        hipMemAllocationGranularityMinimum
        hipMemAllocationGranularityRecommended

    cdef enum hipMemHandleType:
        hipMemHandleTypeGeneric

    cdef enum hipMemOperationType:
        hipMemOperationTypeMap
        hipMemOperationTypeUnmap

    cdef enum hipArraySparseSubresourceType:
        hipArraySparseSubresourceTypeSparseLevel
        hipArraySparseSubresourceTypeMiptail

cdef union hipArrayMapInfo_union_0:
    hipMipmappedArray mipmap
    hipArray_t array

cdef struct hipArrayMapInfo_union_1_struct_0:
    unsigned int level
    unsigned int layer
    unsigned int offsetX
    unsigned int offsetY
    unsigned int offsetZ
    unsigned int extentWidth
    unsigned int extentHeight
    unsigned int extentDepth

cdef struct hipArrayMapInfo_union_1_struct_1:
    unsigned int layer
    unsigned long long offset
    unsigned long long size

cdef union hipArrayMapInfo_union_1:
    hipArrayMapInfo_union_1_struct_0 sparseLevel
    hipArrayMapInfo_union_1_struct_1 miptail

cdef union hipArrayMapInfo_union_2:
    hipMemGenericAllocationHandle_t memHandle

cdef extern from "hip/hip_runtime_api.h":

    cdef struct hipArrayMapInfo:
        hipResourceType resourceType
        hipArrayMapInfo_union_0 resource
        hipArraySparseSubresourceType subresourceType
        hipArrayMapInfo_union_1 subresource
        hipMemOperationType memOperationType
        hipMemHandleType memHandleType
        hipArrayMapInfo_union_2 memHandle
        unsigned long long offset
        unsigned int deviceBitMask
        unsigned int flags
        unsigned int[2] reserved

    # @defgroup API HIP API
    # @{
    # Defines the HIP API.  See the individual sections for more information.
    # @defgroup Driver Initialization and Version
    # @{
    # This section describes the initializtion and version functions of HIP runtime API.
    # @brief Explicitly initializes the HIP runtime.
    # Most HIP APIs implicitly initialize the HIP runtime.
    # This API provides control over the timing of the initialization.
    hipError_t hipInit(unsigned int flags) nogil


    # @brief Returns the approximate HIP driver version.
    # @param [out] driverVersion
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning The HIP feature set does not correspond to an exact CUDA SDK driver revision.
    # This function always set *driverVersion to 4 as an approximation though HIP supports
    # some features which were introduced in later CUDA SDK revisions.
    # HIP apps code should not rely on the driver revision number here and should
    # use arch feature flags to test device capabilities or conditional compilation.
    # @see hipRuntimeGetVersion
    hipError_t hipDriverGetVersion(int * driverVersion) nogil


    # @brief Returns the approximate HIP Runtime version.
    # @param [out] runtimeVersion
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning The version definition of HIP runtime is different from CUDA.
    # On AMD platform, the function returns HIP runtime version,
    # while on NVIDIA platform, it returns CUDA runtime version.
    # And there is no mapping/correlation between HIP version and CUDA version.
    # @see hipDriverGetVersion
    hipError_t hipRuntimeGetVersion(int * runtimeVersion) nogil


    # @brief Returns a handle to a compute device
    # @param [out] device
    # @param [in] ordinal
    # @returns #hipSuccess, #hipErrorInvalidDevice
    hipError_t hipDeviceGet(hipDevice_t * device,int ordinal) nogil


    # @brief Returns the compute capability of the device
    # @param [out] major
    # @param [out] minor
    # @param [in] device
    # @returns #hipSuccess, #hipErrorInvalidDevice
    hipError_t hipDeviceComputeCapability(int * major,int * minor,hipDevice_t device) nogil


    # @brief Returns an identifer string for the device.
    # @param [out] name
    # @param [in] len
    # @param [in] device
    # @returns #hipSuccess, #hipErrorInvalidDevice
    hipError_t hipDeviceGetName(char * name,int len,hipDevice_t device) nogil


    # @brief Returns an UUID for the device.[BETA]
    # @param [out] uuid
    # @param [in] device
    # @beta This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    # @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotInitialized,
    # #hipErrorDeinitialized
    hipError_t hipDeviceGetUuid(hipUUID_t * uuid,hipDevice_t device) nogil


    # @brief Returns a value for attr of link between two devices
    # @param [out] value
    # @param [in] attr
    # @param [in] srcDevice
    # @param [in] dstDevice
    # @returns #hipSuccess, #hipErrorInvalidDevice
    hipError_t hipDeviceGetP2PAttribute(int * value,hipDeviceP2PAttr attr,int srcDevice,int dstDevice) nogil


    # @brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
    # @param [out] pciBusId
    # @param [in] len
    # @param [in] device
    # @returns #hipSuccess, #hipErrorInvalidDevice
    hipError_t hipDeviceGetPCIBusId(char * pciBusId,int len,int device) nogil


    # @brief Returns a handle to a compute device.
    # @param [out] device handle
    # @param [in] PCI Bus ID
    # @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    hipError_t hipDeviceGetByPCIBusId(int * device,const char * pciBusId) nogil


    # @brief Returns the total amount of memory on the device.
    # @param [out] bytes
    # @param [in] device
    # @returns #hipSuccess, #hipErrorInvalidDevice
    hipError_t hipDeviceTotalMem(int * bytes,hipDevice_t device) nogil


    # @}
    # @defgroup Device Device Management
    # @{
    # This section describes the device management functions of HIP runtime API.
    # @brief Waits on all active streams on current device
    # When this command is invoked, the host thread gets blocked until all the commands associated
    # with streams associated with the device. HIP does not support multiple blocking modes (yet!).
    # @returns #hipSuccess
    # @see hipSetDevice, hipDeviceReset
    hipError_t hipDeviceSynchronize() nogil


    # @brief The state of current device is discarded and updated to a fresh state.
    # Calling this function deletes all streams created, memory allocated, kernels running, events
    # created. Make sure that no other thread is using the device or streams, memory, kernels, events
    # associated with the current device.
    # @returns #hipSuccess
    # @see hipDeviceSynchronize
    hipError_t hipDeviceReset() nogil


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
    hipError_t hipSetDevice(int deviceId) nogil


    # @brief Return the default device id for the calling host thread.
    # @param [out] device *device is written with the default device
    # HIP maintains an default device for each thread using thread-local-storage.
    # This device is used implicitly for HIP runtime APIs called by this thread.
    # hipGetDevice returns in * @p device the default device for the calling host thread.
    # @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @see hipSetDevice, hipGetDevicesizeBytes
    hipError_t hipGetDevice(int * deviceId) nogil


    # @brief Return number of compute-capable devices.
    # @param [output] count Returns number of compute-capable devices.
    # @returns #hipSuccess, #hipErrorNoDevice
    # Returns in @p *count the number of devices that have ability to run compute commands.  If there
    # are no such devices, then @ref hipGetDeviceCount will return #hipErrorNoDevice. If 1 or more
    # devices can be found, then hipGetDeviceCount returns #hipSuccess.
    hipError_t hipGetDeviceCount(int * count) nogil


    # @brief Query for a specific device attribute.
    # @param [out] pi pointer to value to return
    # @param [in] attr attribute to query
    # @param [in] deviceId which device to query for information
    # @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    hipError_t hipDeviceGetAttribute(int * pi,hipDeviceAttribute_t attr,int deviceId) nogil


    # @brief Returns the default memory pool of the specified device
    # @param [out] mem_pool Default memory pool to return
    # @param [in] device    Device index for query the default memory pool
    # @returns #chipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotSupported
    # @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    # hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool,int device) nogil


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
    hipError_t hipDeviceSetMemPool(int device,hipMemPool_t mem_pool) nogil


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
    hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool,int device) nogil


    # @brief Returns device properties.
    # @param [out] prop written with device properties
    # @param [in]  deviceId which device to query for information
    # @return #hipSuccess, #hipErrorInvalidDevice
    # @bug HCC always returns 0 for maxThreadsPerMultiProcessor
    # @bug HCC always returns 0 for regsPerBlock
    # @bug HCC always returns 0 for l2CacheSize
    # Populates hipGetDeviceProperties with information for the specified device.
    hipError_t hipGetDeviceProperties(hipDeviceProp_t * prop,int deviceId) nogil


    # @brief Set L1/Shared cache partition.
    # @param [in] cacheConfig
    # @returns #hipSuccess, #hipErrorNotInitialized
    # Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    # on those architectures.
    hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) nogil


    # @brief Get Cache configuration for a specific Device
    # @param [out] cacheConfig
    # @returns #hipSuccess, #hipErrorNotInitialized
    # Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    # on those architectures.
    hipError_t hipDeviceGetCacheConfig(hipFuncCache_t * cacheConfig) nogil


    # @brief Get Resource limits of current device
    # @param [out] pValue
    # @param [in]  limit
    # @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
    # Note: Currently, only hipLimitMallocHeapSize is available
    hipError_t hipDeviceGetLimit(int * pValue,hipLimit_t limit) nogil


    # @brief Set Resource limits of current device
    # @param [in] limit
    # @param [in] value
    # @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
    hipError_t hipDeviceSetLimit(hipLimit_t limit,int value) nogil


    # @brief Returns bank width of shared memory for current device
    # @param [out] pConfig
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    # Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    # ignored on those architectures.
    hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig * pConfig) nogil


    # @brief Gets the flags set for current device
    # @param [out] flags
    # @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    hipError_t hipGetDeviceFlags(unsigned int * flags) nogil


    # @brief The bank width of shared memory on current device is set
    # @param [in] config
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    # Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    # ignored on those architectures.
    hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) nogil


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
    hipError_t hipSetDeviceFlags(unsigned int flags) nogil


    # @brief Device which matches hipDeviceProp_t is returned
    # @param [out] device ID
    # @param [in]  device properties pointer
    # @returns #hipSuccess, #hipErrorInvalidValue
    hipError_t hipChooseDevice(int * device,hipDeviceProp_t * prop) nogil


    # @brief Returns the link type and hop count between two devices
    # @param [in] device1 Ordinal for device1
    # @param [in] device2 Ordinal for device2
    # @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
    # @param [out] hopcount Returns the hop count between the two devices
    # Queries and returns the HSA link type and the hop count between the two specified devices.
    # @returns #hipSuccess, #hipInvalidDevice, #hipErrorRuntimeOther
    hipError_t hipExtGetLinkTypeAndHopCount(int device1,int device2,uint32_t * linktype,uint32_t * hopcount) nogil


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
    hipError_t hipIpcGetMemHandle(hipIpcMemHandle_st * handle,void * devPtr) nogil


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
    hipError_t hipIpcOpenMemHandle(void ** devPtr,hipIpcMemHandle_st handle,unsigned int flags) nogil


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
    hipError_t hipIpcCloseMemHandle(void * devPtr) nogil


    # @brief Gets an opaque interprocess handle for an event.
    # This opaque handle may be copied into other processes and opened with hipIpcOpenEventHandle.
    # Then hipEventRecord, hipEventSynchronize, hipStreamWaitEvent and hipEventQuery may be used in
    # either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
    # will result in undefined behavior.
    # @param[out]  handle Pointer to hipIpcEventHandle to return the opaque event handle
    # @param[in]   event  Event allocated with hipEventInterprocess and hipEventDisableTiming flags
    # @returns #hipSuccess, #hipErrorInvalidConfiguration, #hipErrorInvalidValue
    hipError_t hipIpcGetEventHandle(hipIpcEventHandle_st * handle,hipEvent_t event) nogil


    # @brief Opens an interprocess event handles.
    # Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle. The returned
    # hipEvent_t behaves like a locally created event with the hipEventDisableTiming flag specified. This event
    # need be freed with hipEventDestroy. Operations on the imported event after the exported event has been freed
    # with hipEventDestroy will result in undefined behavior. If the function is called within the same process where
    # handle is returned by hipIpcGetEventHandle, it will return hipErrorInvalidContext.
    # @param[out]  event  Pointer to hipEvent_t to return the event
    # @param[in]   handle The opaque interprocess handle to open
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext
    hipError_t hipIpcOpenEventHandle(hipEvent_t* event,hipIpcEventHandle_st handle) nogil


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
    hipError_t hipFuncSetAttribute(const void * func,hipFuncAttribute attr,int value) nogil


    # @brief Set Cache configuration for a specific function
    # @param [in] config;
    # @returns #hipSuccess, #hipErrorNotInitialized
    # Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    # on those architectures.
    hipError_t hipFuncSetCacheConfig(const void * func,hipFuncCache_t config) nogil


    # @brief Set shared memory configuation for a specific function
    # @param [in] func
    # @param [in] config
    # @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
    # Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    # ignored on those architectures.
    hipError_t hipFuncSetSharedMemConfig(const void * func,hipSharedMemConfig config) nogil


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
    hipError_t hipGetLastError() nogil


    # @brief Return last error returned by any HIP runtime API call.
    # @return #hipSuccess
    # Returns the last error that has been returned by any of the runtime calls in the same host
    # thread. Unlike hipGetLastError, this function does not reset the saved error code.
    # @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    hipError_t hipPeekAtLastError() nogil


    # @brief Return hip error as text string form.
    # @param hip_error Error code to convert to name.
    # @return const char pointer to the NULL-terminated error name
    # @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    const char * hipGetErrorName(hipError_t hip_error) nogil


    # @brief Return handy text string message to explain the error which occurred
    # @param hipError Error code to convert to string.
    # @return const char pointer to the NULL-terminated error string
    # @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    const char * hipGetErrorString(hipError_t hipError) nogil


    # @brief Return hip error as text string form.
    # @param [in] hipError Error code to convert to string.
    # @param [out] const char pointer to the NULL-terminated error string
    # @return #hipSuccess, #hipErrorInvalidValue
    # @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    hipError_t hipDrvGetErrorName(hipError_t hipError,const char ** errorString) nogil


    # @brief Return handy text string message to explain the error which occurred
    # @param [in] hipError Error code to convert to string.
    # @param [out] const char pointer to the NULL-terminated error string
    # @return #hipSuccess, #hipErrorInvalidValue
    # @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    hipError_t hipDrvGetErrorString(hipError_t hipError,const char ** errorString) nogil


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
    hipError_t hipStreamCreate(hipStream_t* stream) nogil


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
    hipError_t hipStreamCreateWithFlags(hipStream_t* stream,unsigned int flags) nogil


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
    hipError_t hipStreamCreateWithPriority(hipStream_t* stream,unsigned int flags,int priority) nogil


    # @brief Returns numerical values that correspond to the least and greatest stream priority.
    # @param[in, out] leastPriority pointer in which value corresponding to least priority is returned.
    # @param[in, out] greatestPriority pointer in which value corresponding to greatest priority is returned.
    # Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least
    # and greatest stream priority respectively. Stream priorities follow a convention where lower numbers
    # imply greater priorities. The range of meaningful stream priorities is given by
    # [*greatestPriority, *leastPriority]. If the user attempts to create a stream with a priority value
    # that is outside the the meaningful range as specified by this API, the priority is automatically
    # clamped to within the valid range.
    hipError_t hipDeviceGetStreamPriorityRange(int * leastPriority,int * greatestPriority) nogil


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
    hipError_t hipStreamDestroy(hipStream_t stream) nogil


    # @brief Return #hipSuccess if all of the operations in the specified @p stream have completed, or
    # #hipErrorNotReady if not.
    # @param[in] stream stream to query
    # @return #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle
    # This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
    # host threads are sending work to the stream, the status may change immediately after the function
    # is called.  It is typically used for debug.
    # @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamSynchronize,
    # hipStreamDestroy
    hipError_t hipStreamQuery(hipStream_t stream) nogil


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
    hipError_t hipStreamSynchronize(hipStream_t stream) nogil


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
    hipError_t hipStreamWaitEvent(hipStream_t stream,hipEvent_t event,unsigned int flags) nogil


    # @brief Return flags associated with this stream.
    # @param[in] stream stream to be queried
    # @param[in,out] flags Pointer to an unsigned integer in which the stream's flags are returned
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
    # @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
    # Return flags associated with this stream in *@p flags.
    # @see hipStreamCreateWithFlags
    hipError_t hipStreamGetFlags(hipStream_t stream,unsigned int * flags) nogil


    # @brief Query the priority of a stream.
    # @param[in] stream stream to be queried
    # @param[in,out] priority Pointer to an unsigned integer in which the stream's priority is returned
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
    # @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
    # Query the priority of a stream. The priority is returned in in priority.
    # @see hipStreamCreateWithFlags
    hipError_t hipStreamGetPriority(hipStream_t stream,int * priority) nogil


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
    hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream,uint32_t cuMaskSize,uint32_t * cuMask) nogil


    # @brief Get CU mask associated with an asynchronous stream
    # @param[in] stream stream to be queried
    # @param[in] cuMaskSize number of the block of memories (uint32_t *) allocated by user
    # @param[out] cuMask Pointer to a pre-allocated block of memories (uint32_t *) in which
    # the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
    # each active bit represents one active CU
    # @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
    # @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    hipError_t hipExtStreamGetCUMask(hipStream_t stream,uint32_t cuMaskSize,uint32_t * cuMask) nogil


    ctypedef void (*hipStreamCallback_t) (hipStream_t,hipError_t,void *)

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
    hipError_t hipStreamAddCallback(hipStream_t stream,hipStreamCallback_t callback,void * userData,unsigned int flags) nogil


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
    hipError_t hipStreamWaitValue32(hipStream_t stream,void * ptr,uint32_t value,unsigned int flags,uint32_t mask) nogil


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
    hipError_t hipStreamWaitValue64(hipStream_t stream,void * ptr,uint64_t value,unsigned int flags,uint64_t mask) nogil


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
    hipError_t hipStreamWriteValue32(hipStream_t stream,void * ptr,uint32_t value,unsigned int flags) nogil


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
    hipError_t hipStreamWriteValue64(hipStream_t stream,void * ptr,uint64_t value,unsigned int flags) nogil


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
    hipError_t hipEventCreateWithFlags(hipEvent_t* event,unsigned int flags) nogil


    # Create an event
    # @param[in,out] event Returns the newly created event.
    # @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
    # #hipErrorLaunchFailure, #hipErrorOutOfMemory
    # @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize,
    # hipEventDestroy, hipEventElapsedTime
    hipError_t hipEventCreate(hipEvent_t* event) nogil



    hipError_t hipEventRecord(hipEvent_t event,hipStream_t stream) nogil


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
    hipError_t hipEventDestroy(hipEvent_t event) nogil


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
    hipError_t hipEventSynchronize(hipEvent_t event) nogil


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
    hipError_t hipEventElapsedTime(float * ms,hipEvent_t start,hipEvent_t stop) nogil


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
    hipError_t hipEventQuery(hipEvent_t event) nogil


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
    hipError_t hipPointerGetAttributes(hipPointerAttribute_t * attributes,const void * ptr) nogil


    # @brief Returns information about the specified pointer.[BETA]
    # @param [in, out] data     returned pointer attribute value
    # @param [in]      atribute attribute to query for
    # @param [in]      ptr      pointer to get attributes for
    # @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @beta This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    # @see hipPointerGetAttributes
    hipError_t hipPointerGetAttribute(void * data,hipPointer_attribute attribute,hipDeviceptr_t ptr) nogil


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
    hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes,hipPointer_attribute * attributes,void ** data,hipDeviceptr_t ptr) nogil


    # @brief Imports an external semaphore.
    # @param[out] extSem_out  External semaphores to be waited on
    # @param[in] semHandleDesc Semaphore import handle descriptor
    # @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @see
    hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,hipExternalSemaphoreHandleDesc_st * semHandleDesc) nogil


    # @brief Signals a set of external semaphore objects.
    # @param[in] extSem_out  External semaphores to be waited on
    # @param[in] paramsArray Array of semaphore parameters
    # @param[in] numExtSems Number of semaphores to wait on
    # @param[in] stream Stream to enqueue the wait operations in
    # @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @see
    hipError_t hipSignalExternalSemaphoresAsync(hipExternalSemaphore_t * extSemArray,hipExternalSemaphoreSignalParams_st * paramsArray,unsigned int numExtSems,hipStream_t stream) nogil


    # @brief Waits on a set of external semaphore objects
    # @param[in] extSem_out  External semaphores to be waited on
    # @param[in] paramsArray Array of semaphore parameters
    # @param[in] numExtSems Number of semaphores to wait on
    # @param[in] stream Stream to enqueue the wait operations in
    # @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @see
    hipError_t hipWaitExternalSemaphoresAsync(hipExternalSemaphore_t * extSemArray,hipExternalSemaphoreWaitParams_st * paramsArray,unsigned int numExtSems,hipStream_t stream) nogil


    # @brief Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.
    # @param[in] extSem handle to an external memory object
    # @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @see
    hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) nogil


    # @brief Imports an external memory object.
    # @param[out] extMem_out  Returned handle to an external memory object
    # @param[in]  memHandleDesc Memory import handle descriptor
    # @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @see
    hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out,hipExternalMemoryHandleDesc_st * memHandleDesc) nogil


    # @brief Maps a buffer onto an imported memory object.
    # @param[out] devPtr Returned device pointer to buffer
    # @param[in]  extMem  Handle to external memory object
    # @param[in]  bufferDesc  Buffer descriptor
    # @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @see
    hipError_t hipExternalMemoryGetMappedBuffer(void ** devPtr,hipExternalMemory_t extMem,hipExternalMemoryBufferDesc_st * bufferDesc) nogil


    # @brief Destroys an external memory object.
    # @param[in] extMem  External memory object to be destroyed
    # @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @see
    hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) nogil


    # @brief Allocate memory on the default accelerator
    # @param[out] ptr Pointer to the allocated memory
    # @param[in]  size Requested memory size
    # If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    # @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
    # @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
    # hipHostFree, hipHostMalloc
    hipError_t hipMalloc(void ** ptr,int size) nogil


    # @brief Allocate memory on the default accelerator
    # @param[out] ptr Pointer to the allocated memory
    # @param[in]  size Requested memory size
    # @param[in]  flags Type of memory allocation
    # If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    # @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
    # @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
    # hipHostFree, hipHostMalloc
    hipError_t hipExtMallocWithFlags(void ** ptr,int sizeBytes,unsigned int flags) nogil


    # @brief Allocate pinned host memory [Deprecated]
    # @param[out] ptr Pointer to the allocated host pinned memory
    # @param[in]  size Requested memory size
    # If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    # @return #hipSuccess, #hipErrorOutOfMemory
    # @deprecated use hipHostMalloc() instead
    hipError_t hipMallocHost(void ** ptr,int size) nogil


    # @brief Allocate pinned host memory [Deprecated]
    # @param[out] ptr Pointer to the allocated host pinned memory
    # @param[in]  size Requested memory size
    # If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    # @return #hipSuccess, #hipErrorOutOfMemory
    # @deprecated use hipHostMalloc() instead
    hipError_t hipMemAllocHost(void ** ptr,int size) nogil


    # @brief Allocate device accessible page locked host memory
    # @param[out] ptr Pointer to the allocated host pinned memory
    # @param[in]  size Requested memory size
    # @param[in]  flags Type of host memory allocation
    # If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    # @return #hipSuccess, #hipErrorOutOfMemory
    # @see hipSetDeviceFlags, hipHostFree
    hipError_t hipHostMalloc(void ** ptr,int size,unsigned int flags) nogil


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
    hipError_t hipMallocManaged(void ** dev_ptr,int size,unsigned int flags) nogil


    # @brief Prefetches memory to the specified destination device using HIP.
    # @param [in] dev_ptr  pointer to be prefetched
    # @param [in] count    size in bytes for prefetching
    # @param [in] device   destination device to prefetch to
    # @param [in] stream   stream to enqueue prefetch operation
    # @returns #hipSuccess, #hipErrorInvalidValue
    hipError_t hipMemPrefetchAsync(const void * dev_ptr,int count,int device,hipStream_t stream) nogil


    # @brief Advise about the usage of a given memory range to HIP.
    # @param [in] dev_ptr  pointer to memory to set the advice for
    # @param [in] count    size in bytes of the memory range
    # @param [in] advice   advice to be applied for the specified memory range
    # @param [in] device   device to apply the advice for
    # @returns #hipSuccess, #hipErrorInvalidValue
    hipError_t hipMemAdvise(const void * dev_ptr,int count,hipMemoryAdvise advice,int device) nogil


    # @brief Query an attribute of a given memory range in HIP.
    # @param [in,out] data   a pointer to a memory location where the result of each
    # attribute query will be written to
    # @param [in] data_size  the size of data
    # @param [in] attribute  the attribute to query
    # @param [in] dev_ptr    start of the range to query
    # @param [in] count      size of the range to query
    # @returns #hipSuccess, #hipErrorInvalidValue
    hipError_t hipMemRangeGetAttribute(void * data,int data_size,hipMemRangeAttribute attribute,const void * dev_ptr,int count) nogil


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
    hipError_t hipMemRangeGetAttributes(void ** data,int * data_sizes,hipMemRangeAttribute * attributes,int num_attributes,const void * dev_ptr,int count) nogil


    # @brief Attach memory to a stream asynchronously in HIP.
    # @param [in] stream     - stream in which to enqueue the attach operation
    # @param [in] dev_ptr    - pointer to memory (must be a pointer to managed memory or
    # to a valid host-accessible region of system-allocated memory)
    # @param [in] length     - length of memory (defaults to zero)
    # @param [in] flags      - must be one of hipMemAttachGlobal, hipMemAttachHost or
    # hipMemAttachSingle (defaults to hipMemAttachSingle)
    # @returns #hipSuccess, #hipErrorInvalidValue
    hipError_t hipStreamAttachMemAsync(hipStream_t stream,void * dev_ptr,int length,unsigned int flags) nogil


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
    hipError_t hipMallocAsync(void ** dev_ptr,int size,hipStream_t stream) nogil


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
    hipError_t hipFreeAsync(void * dev_ptr,hipStream_t stream) nogil


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
    hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool,int min_bytes_to_hold) nogil


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
    hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool,hipMemPoolAttr attr,void * value) nogil


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
    hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool,hipMemPoolAttr attr,void * value) nogil


    # @brief Controls visibility of the specified pool between devices
    # @param [in] mem_pool   Memory pool for acccess change
    # @param [in] desc_list  Array of access descriptors. Each descriptor instructs the access to enable for a single gpu
    # @param [in] count  Number of descriptors in the map array.
    # @returns  #hipSuccess, #hipErrorInvalidValue
    # @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
    # hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolGetAccess
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool,hipMemAccessDesc * desc_list,int count) nogil


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
    hipError_t hipMemPoolGetAccess(hipMemAccessFlags * flags,hipMemPool_t mem_pool,hipMemLocation * location) nogil


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
    hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool,hipMemPoolProps * pool_props) nogil


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
    hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) nogil


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
    hipError_t hipMallocFromPoolAsync(void ** dev_ptr,int size,hipMemPool_t mem_pool,hipStream_t stream) nogil


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
    hipError_t hipMemPoolExportToShareableHandle(void * shared_handle,hipMemPool_t mem_pool,hipMemAllocationHandleType handle_type,unsigned int flags) nogil


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
    hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool,void * shared_handle,hipMemAllocationHandleType handle_type,unsigned int flags) nogil


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
    hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData * export_data,void * dev_ptr) nogil


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
    hipError_t hipMemPoolImportPointer(void ** dev_ptr,hipMemPool_t mem_pool,hipMemPoolPtrExportData * export_data) nogil


    # @brief Allocate device accessible page locked host memory [Deprecated]
    # @param[out] ptr Pointer to the allocated host pinned memory
    # @param[in]  size Requested memory size
    # @param[in]  flags Type of host memory allocation
    # If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    # @return #hipSuccess, #hipErrorOutOfMemory
    # @deprecated use hipHostMalloc() instead
    hipError_t hipHostAlloc(void ** ptr,int size,unsigned int flags) nogil


    # @brief Get Device pointer from Host Pointer allocated through hipHostMalloc
    # @param[out] dstPtr Device Pointer mapped to passed host pointer
    # @param[in]  hstPtr Host Pointer allocated through hipHostMalloc
    # @param[in]  flags Flags to be passed for extension
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    # @see hipSetDeviceFlags, hipHostMalloc
    hipError_t hipHostGetDevicePointer(void ** devPtr,void * hstPtr,unsigned int flags) nogil


    # @brief Return flags associated with host pointer
    # @param[out] flagsPtr Memory location to store flags
    # @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
    # @return #hipSuccess, #hipErrorInvalidValue
    # @see hipHostMalloc
    hipError_t hipHostGetFlags(unsigned int * flagsPtr,void * hostPtr) nogil


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
    hipError_t hipHostRegister(void * hostPtr,int sizeBytes,unsigned int flags) nogil


    # @brief Un-register host pointer
    # @param[in] hostPtr Host pointer previously registered with #hipHostRegister
    # @return Error code
    # @see hipHostRegister
    hipError_t hipHostUnregister(void * hostPtr) nogil


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
    hipError_t hipMallocPitch(void ** ptr,int * pitch,int width,int height) nogil


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
    hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr,int * pitch,int widthInBytes,int height,unsigned int elementSizeBytes) nogil


    # @brief Free memory allocated by the hcc hip memory allocation API.
    # This API performs an implicit hipDeviceSynchronize() call.
    # If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
    # @param[in] ptr Pointer to memory to be freed
    # @return #hipSuccess
    # @return #hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated
    # with hipHostMalloc)
    # @see hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
    # hipMalloc3DArray, hipHostMalloc
    hipError_t hipFree(void * ptr) nogil


    # @brief Free memory allocated by the hcc hip host memory allocation API.  [Deprecated]
    # @param[in] ptr Pointer to memory to be freed
    # @return #hipSuccess,
    # #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
    #  hipMalloc)
    # @deprecated use hipHostFree() instead
    hipError_t hipFreeHost(void * ptr) nogil


    # @brief Free memory allocated by the hcc hip host memory allocation API
    # This API performs an implicit hipDeviceSynchronize() call.
    # If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
    # @param[in] ptr Pointer to memory to be freed
    # @return #hipSuccess,
    # #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
    # hipMalloc)
    # @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D,
    # hipMalloc3DArray, hipHostMalloc
    hipError_t hipHostFree(void * ptr) nogil


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
    hipError_t hipMemcpy(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind) nogil



    hipError_t hipMemcpyWithStream(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil


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
    hipError_t hipMemcpyHtoD(hipDeviceptr_t dst,void * src,int sizeBytes) nogil


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
    hipError_t hipMemcpyDtoH(void * dst,hipDeviceptr_t src,int sizeBytes) nogil


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
    hipError_t hipMemcpyDtoD(hipDeviceptr_t dst,hipDeviceptr_t src,int sizeBytes) nogil


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
    hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst,void * src,int sizeBytes,hipStream_t stream) nogil


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
    hipError_t hipMemcpyDtoHAsync(void * dst,hipDeviceptr_t src,int sizeBytes,hipStream_t stream) nogil


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
    hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst,hipDeviceptr_t src,int sizeBytes,hipStream_t stream) nogil


    # @brief Returns a global pointer from a module.
    # Returns in *dptr and *bytes the pointer and size of the global of name name located in module hmod.
    # If no variable of that name exists, it returns hipErrorNotFound. Both parameters dptr and bytes are optional.
    # If one of them is NULL, it is ignored and hipSuccess is returned.
    # @param[out]  dptr  Returns global device pointer
    # @param[out]  bytes Returns global size in bytes
    # @param[in]   hmod  Module to retrieve global from
    # @param[in]   name  Name of global to retrieve
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotFound, #hipErrorInvalidContext
    hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr,int * bytes,hipModule_t hmod,const char * name) nogil


    # @brief Gets device pointer associated with symbol on the device.
    # @param[out]  devPtr  pointer to the device associated the symbole
    # @param[in]   symbol  pointer to the symbole of the device
    # @return #hipSuccess, #hipErrorInvalidValue
    hipError_t hipGetSymbolAddress(void ** devPtr,const void * symbol) nogil


    # @brief Gets the size of the given symbol on the device.
    # @param[in]   symbol  pointer to the device symbole
    # @param[out]  size  pointer to the size
    # @return #hipSuccess, #hipErrorInvalidValue
    hipError_t hipGetSymbolSize(int * size,const void * symbol) nogil


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
    hipError_t hipMemcpyToSymbol(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind) nogil


    # @brief Copies data to the given symbol on the device asynchronously.
    # @param[out]  symbol  pointer to the device symbole
    # @param[in]   src  pointer to the source address
    # @param[in]   sizeBytes  size in bytes to copy
    # @param[in]   offset  offset in bytes from start of symbole
    # @param[in]   kind  type of memory transfer
    # @param[in]   stream  stream identifier
    # @return #hipSuccess, #hipErrorInvalidValue
    hipError_t hipMemcpyToSymbolAsync(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil


    # @brief Copies data from the given symbol on the device.
    # @param[out]  dptr  Returns pointer to destinition memory address
    # @param[in]   symbol  pointer to the symbole address on the device
    # @param[in]   sizeBytes  size in bytes to copy
    # @param[in]   offset  offset in bytes from the start of symbole
    # @param[in]   kind  type of memory transfer
    # @return #hipSuccess, #hipErrorInvalidValue
    hipError_t hipMemcpyFromSymbol(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind) nogil


    # @brief Copies data from the given symbol on the device asynchronously.
    # @param[out]  dptr  Returns pointer to destinition memory address
    # @param[in]   symbol  pointer to the symbole address on the device
    # @param[in]   sizeBytes  size in bytes to copy
    # @param[in]   offset  offset in bytes from the start of symbole
    # @param[in]   kind  type of memory transfer
    # @param[in]   stream  stream identifier
    # @return #hipSuccess, #hipErrorInvalidValue
    hipError_t hipMemcpyFromSymbolAsync(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil


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
    hipError_t hipMemcpyAsync(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil


    # @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    # byte value value.
    # @param[out] dst Data being filled
    # @param[in]  constant value to be set
    # @param[in]  sizeBytes Data size in bytes
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    hipError_t hipMemset(void * dst,int value,int sizeBytes) nogil


    # @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    # byte value value.
    # @param[out] dst Data ptr to be filled
    # @param[in]  constant value to be set
    # @param[in]  number of values to be set
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    hipError_t hipMemsetD8(hipDeviceptr_t dest,unsigned char value,int count) nogil


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
    hipError_t hipMemsetD8Async(hipDeviceptr_t dest,unsigned char value,int count,hipStream_t stream) nogil


    # @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    # short value value.
    # @param[out] dst Data ptr to be filled
    # @param[in]  constant value to be set
    # @param[in]  number of values to be set
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    hipError_t hipMemsetD16(hipDeviceptr_t dest,unsigned short value,int count) nogil


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
    hipError_t hipMemsetD16Async(hipDeviceptr_t dest,unsigned short value,int count,hipStream_t stream) nogil


    # @brief Fills the memory area pointed to by dest with the constant integer
    # value for specified number of times.
    # @param[out] dst Data being filled
    # @param[in]  constant value to be set
    # @param[in]  number of values to be set
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    hipError_t hipMemsetD32(hipDeviceptr_t dest,int value,int count) nogil


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
    hipError_t hipMemsetAsync(void * dst,int value,int sizeBytes,hipStream_t stream) nogil


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
    hipError_t hipMemsetD32Async(hipDeviceptr_t dst,int value,int count,hipStream_t stream) nogil


    # @brief Fills the memory area pointed to by dst with the constant value.
    # @param[out] dst Pointer to device memory
    # @param[in]  pitch - data size in bytes
    # @param[in]  value - constant value to be set
    # @param[in]  width
    # @param[in]  height
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    hipError_t hipMemset2D(void * dst,int pitch,int value,int width,int height) nogil


    # @brief Fills asynchronously the memory area pointed to by dst with the constant value.
    # @param[in]  dst Pointer to device memory
    # @param[in]  pitch - data size in bytes
    # @param[in]  value - constant value to be set
    # @param[in]  width
    # @param[in]  height
    # @param[in]  stream
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    hipError_t hipMemset2DAsync(void * dst,int pitch,int value,int width,int height,hipStream_t stream) nogil


    # @brief Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
    # @param[in] pitchedDevPtr
    # @param[in]  value - constant value to be set
    # @param[in]  extent
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent) nogil


    # @brief Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
    # @param[in] pitchedDevPtr
    # @param[in]  value - constant value to be set
    # @param[in]  extent
    # @param[in]  stream
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent,hipStream_t stream) nogil


    # @brief Query memory info.
    # Return snapshot of free memory, and total allocatable memory on the device.
    # Returns in *free a snapshot of the current free memory.
    # @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    # @warning On HCC, the free memory only accounts for memory allocated by this process and may be
    # optimistic.
    hipError_t hipMemGetInfo(int * free,int * total) nogil



    hipError_t hipMemPtrGetInfo(void * ptr,int * size) nogil


    # @brief Allocate an array on the device.
    # @param[out]  array  Pointer to allocated array in device memory
    # @param[in]   desc   Requested channel format
    # @param[in]   width  Requested array allocation width
    # @param[in]   height Requested array allocation height
    # @param[in]   flags  Requested properties of allocated array
    # @return      #hipSuccess, #hipErrorOutOfMemory
    # @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
    hipError_t hipMallocArray(hipArray ** array,hipChannelFormatDesc * desc,int width,int height,unsigned int flags) nogil



    hipError_t hipArrayCreate(hipArray ** pHandle,HIP_ARRAY_DESCRIPTOR * pAllocateArray) nogil



    hipError_t hipArrayDestroy(hipArray * array) nogil



    hipError_t hipArray3DCreate(hipArray ** array,HIP_ARRAY3D_DESCRIPTOR * pAllocateArray) nogil



    hipError_t hipMalloc3D(hipPitchedPtr * pitchedDevPtr,hipExtent extent) nogil


    # @brief Frees an array on the device.
    # @param[in]  array  Pointer to array to free
    # @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    # @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
    hipError_t hipFreeArray(hipArray * array) nogil


    # @brief Frees a mipmapped array on the device
    # @param[in] mipmappedArray - Pointer to mipmapped array to free
    # @return #hipSuccess, #hipErrorInvalidValue
    hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) nogil


    # @brief Allocate an array on the device.
    # @param[out]  array  Pointer to allocated array in device memory
    # @param[in]   desc   Requested channel format
    # @param[in]   extent Requested array allocation width, height and depth
    # @param[in]   flags  Requested properties of allocated array
    # @return      #hipSuccess, #hipErrorOutOfMemory
    # @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
    hipError_t hipMalloc3DArray(hipArray ** array,hipChannelFormatDesc * desc,hipExtent extent,unsigned int flags) nogil


    # @brief Allocate a mipmapped array on the device
    # @param[out] mipmappedArray  - Pointer to allocated mipmapped array in device memory
    # @param[in]  desc            - Requested channel format
    # @param[in]  extent          - Requested allocation size (width field in elements)
    # @param[in]  numLevels       - Number of mipmap levels to allocate
    # @param[in]  flags           - Flags for extensions
    # @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray,hipChannelFormatDesc * desc,hipExtent extent,unsigned int numLevels,unsigned int flags) nogil


    # @brief Gets a mipmap level of a HIP mipmapped array
    # @param[out] levelArray     - Returned mipmap level HIP array
    # @param[in]  mipmappedArray - HIP mipmapped array
    # @param[in]  level          - Mipmap level
    # @return #hipSuccess, #hipErrorInvalidValue
    hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray,hipMipmappedArray_const_t mipmappedArray,unsigned int level) nogil


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
    hipError_t hipMemcpy2D(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil


    # @brief Copies memory for 2D arrays.
    # @param[in]   pCopy Parameters for the memory copy
    # @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    # #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    # @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    # hipMemcpyToSymbol, hipMemcpyAsync
    hipError_t hipMemcpyParam2D(hip_Memcpy2D * pCopy) nogil


    # @brief Copies memory for 2D arrays.
    # @param[in]   pCopy Parameters for the memory copy
    # @param[in]   stream Stream to use
    # @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    # #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    # @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    # hipMemcpyToSymbol, hipMemcpyAsync
    hipError_t hipMemcpyParam2DAsync(hip_Memcpy2D * pCopy,hipStream_t stream) nogil


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
    hipError_t hipMemcpy2DAsync(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil


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
    hipError_t hipMemcpy2DToArray(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil


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
    hipError_t hipMemcpy2DToArrayAsync(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil


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
    hipError_t hipMemcpyToArray(hipArray * dst,int wOffset,int hOffset,const void * src,int count,hipMemcpyKind kind) nogil


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
    hipError_t hipMemcpyFromArray(void * dst,hipArray_const_t srcArray,int wOffset,int hOffset,int count,hipMemcpyKind kind) nogil


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
    hipError_t hipMemcpy2DFromArray(void * dst,int dpitch,hipArray_const_t src,int wOffset,int hOffset,int width,int height,hipMemcpyKind kind) nogil


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
    hipError_t hipMemcpy2DFromArrayAsync(void * dst,int dpitch,hipArray_const_t src,int wOffset,int hOffset,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil


    # @brief Copies data between host and device.
    # @param[in]   dst       Destination memory address
    # @param[in]   srcArray  Source array
    # @param[in]   srcoffset Offset in bytes of source array
    # @param[in]   count     Size of memory copy in bytes
    # @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    # #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    # @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    # hipMemcpyAsync
    hipError_t hipMemcpyAtoH(void * dst,hipArray * srcArray,int srcOffset,int count) nogil


    # @brief Copies data between host and device.
    # @param[in]   dstArray   Destination memory address
    # @param[in]   dstOffset  Offset in bytes of destination array
    # @param[in]   srcHost    Source host pointer
    # @param[in]   count      Size of memory copy in bytes
    # @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    # #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    # @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    # hipMemcpyAsync
    hipError_t hipMemcpyHtoA(hipArray * dstArray,int dstOffset,const void * srcHost,int count) nogil


    # @brief Copies data between host and device.
    # @param[in]   p   3D memory copy parameters
    # @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    # #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    # @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    # hipMemcpyAsync
    hipError_t hipMemcpy3D(hipMemcpy3DParms * p) nogil


    # @brief Copies data between host and device asynchronously.
    # @param[in]   p        3D memory copy parameters
    # @param[in]   stream   Stream to use
    # @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    # #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    # @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    # hipMemcpyAsync
    hipError_t hipMemcpy3DAsync(hipMemcpy3DParms * p,hipStream_t stream) nogil


    # @brief Copies data between host and device.
    # @param[in]   pCopy   3D memory copy parameters
    # @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    # #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    # @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    # hipMemcpyAsync
    hipError_t hipDrvMemcpy3D(HIP_MEMCPY3D * pCopy) nogil


    # @brief Copies data between host and device asynchronously.
    # @param[in]   pCopy    3D memory copy parameters
    # @param[in]   stream   Stream to use
    # @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    # #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    # @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    # hipMemcpyAsync
    hipError_t hipDrvMemcpy3DAsync(HIP_MEMCPY3D * pCopy,hipStream_t stream) nogil


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
    hipError_t hipDeviceCanAccessPeer(int * canAccessPeer,int deviceId,int peerDeviceId) nogil


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
    hipError_t hipDeviceEnablePeerAccess(int peerDeviceId,unsigned int flags) nogil


    # @brief Disable direct access from current device's virtual address space to memory allocations
    # physically located on a peer device.
    # Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
    # enabled from the current device.
    # @param [in] peerDeviceId
    # @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
    hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) nogil


    # @brief Get information on memory allocations.
    # @param [out] pbase - BAse pointer address
    # @param [out] psize - Size of allocation
    # @param [in]  dptr- Device Pointer
    # @returns #hipSuccess, #hipErrorInvalidDevicePointer
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase,int * psize,hipDeviceptr_t dptr) nogil


    # @brief Copies memory from one device to memory on another device.
    # @param [out] dst - Destination device pointer.
    # @param [in] dstDeviceId - Destination device
    # @param [in] src - Source device pointer
    # @param [in] srcDeviceId - Source device
    # @param [in] sizeBytes - Size of memory copy in bytes
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
    hipError_t hipMemcpyPeer(void * dst,int dstDeviceId,const void * src,int srcDeviceId,int sizeBytes) nogil


    # @brief Copies memory from one device to memory on another device.
    # @param [out] dst - Destination device pointer.
    # @param [in] dstDevice - Destination device
    # @param [in] src - Source device pointer
    # @param [in] srcDevice - Source device
    # @param [in] sizeBytes - Size of memory copy in bytes
    # @param [in] stream - Stream identifier
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
    hipError_t hipMemcpyPeerAsync(void * dst,int dstDeviceId,const void * src,int srcDevice,int sizeBytes,hipStream_t stream) nogil


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
    hipError_t hipCtxCreate(hipCtx_t* ctx,unsigned int flags,hipDevice_t device) nogil


    # @brief Destroy a HIP context.
    # @param [in] ctx Context to destroy
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @see hipCtxCreate, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,hipCtxSetCurrent,
    # hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    hipError_t hipCtxDestroy(hipCtx_t ctx) nogil


    # @brief Pop the current/default context and return the popped context.
    # @param [out] ctx
    # @returns #hipSuccess, #hipErrorInvalidContext
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxSetCurrent, hipCtxGetCurrent,
    # hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipCtxPopCurrent(hipCtx_t* ctx) nogil


    # @brief Push the context to be set as current/ default context
    # @param [in] ctx
    # @returns #hipSuccess, #hipErrorInvalidContext
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    hipError_t hipCtxPushCurrent(hipCtx_t ctx) nogil


    # @brief Set the passed context as current/default
    # @param [in] ctx
    # @returns #hipSuccess, #hipErrorInvalidContext
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    hipError_t hipCtxSetCurrent(hipCtx_t ctx) nogil


    # @brief Get the handle of the current/ default context
    # @param [out] ctx
    # @returns #hipSuccess, #hipErrorInvalidContext
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
    # hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipCtxGetCurrent(hipCtx_t* ctx) nogil


    # @brief Get the handle of the device associated with current/default context
    # @param [out] device
    # @returns #hipSuccess, #hipErrorInvalidContext
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize
    hipError_t hipCtxGetDevice(hipDevice_t * device) nogil


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
    hipError_t hipCtxGetApiVersion(hipCtx_t ctx,int * apiVersion) nogil


    # @brief Set Cache configuration for a specific function
    # @param [out] cacheConfiguration
    # @return #hipSuccess
    # @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
    # ignored on those architectures.
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipCtxGetCacheConfig(hipFuncCache_t * cacheConfig) nogil


    # @brief Set L1/Shared cache partition.
    # @param [in] cacheConfiguration
    # @return #hipSuccess
    # @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
    # ignored on those architectures.
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) nogil


    # @brief Set Shared memory bank configuration.
    # @param [in] sharedMemoryConfiguration
    # @return #hipSuccess
    # @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    # ignored on those architectures.
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) nogil


    # @brief Get Shared memory bank configuration.
    # @param [out] sharedMemoryConfiguration
    # @return #hipSuccess
    # @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    # ignored on those architectures.
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig * pConfig) nogil


    # @brief Blocks until the default context has completed all preceding requested tasks.
    # @return #hipSuccess
    # @warning This function waits for all streams on the default context to complete execution, and
    # then returns.
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxGetDevice
    hipError_t hipCtxSynchronize() nogil


    # @brief Return flags used for creating default context.
    # @param [out] flags
    # @returns #hipSuccess
    # @see hipCtxCreate, hipCtxDestroy, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipCtxGetFlags(unsigned int * flags) nogil


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
    hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx,unsigned int flags) nogil


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
    hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) nogil


    # @}
    # @brief Get the state of the primary context.
    # @param [in] Device to get primary context flags for
    # @param [out] Pointer to store flags
    # @param [out] Pointer to store context state; 0 = inactive, 1 = active
    # @returns #hipSuccess
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev,unsigned int * flags,int * active) nogil


    # @brief Release the primary context on the GPU.
    # @param [in] Device which primary context is released
    # @returns #hipSuccess
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    # @warning This function return #hipSuccess though doesn't release the primaryCtx by design on
    # HIP/HCC path.
    hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) nogil


    # @brief Retain the primary context on the GPU.
    # @param [out] Returned context handle of the new context
    # @param [in] Device which primary context is released
    # @returns #hipSuccess
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx,hipDevice_t dev) nogil


    # @brief Resets the primary context on the GPU.
    # @param [in] Device which primary context is reset
    # @returns #hipSuccess
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) nogil


    # @brief Set flags for the primary context.
    # @param [in] Device for which the primary context flags are set
    # @param [in] New flags for the device
    # @returns #hipSuccess, #hipErrorContextAlreadyInUse
    # @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    # hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev,unsigned int flags) nogil


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
    hipError_t hipModuleLoad(hipModule_t* module,const char * fname) nogil


    # @brief Frees the module
    # @param [in] module
    # @returns hipSuccess, hipInvalidValue
    # module is freed and the code objects associated with it are destroyed
    hipError_t hipModuleUnload(hipModule_t module) nogil


    # @brief Function with kname will be extracted if present in module
    # @param [in] module
    # @param [in] kname
    # @param [out] function
    # @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorNotInitialized,
    # hipErrorNotFound,
    hipError_t hipModuleGetFunction(hipFunction_t* function,hipModule_t module,const char * kname) nogil


    # @brief Find out attributes for a given function.
    # @param [out] attr
    # @param [in] func
    # @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
    hipError_t hipFuncGetAttributes(hipFuncAttributes * attr,const void * func) nogil


    # @brief Find out a specific attribute for a given function.
    # @param [out] value
    # @param [in]  attrib
    # @param [in]  hfunc
    # @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
    hipError_t hipFuncGetAttribute(int * value,hipFunction_attribute attrib,hipFunction_t hfunc) nogil


    # @brief returns the handle of the texture reference with the name from the module.
    # @param [in] hmod
    # @param [in] name
    # @param [out] texRef
    # @returns hipSuccess, hipErrorNotInitialized, hipErrorNotFound, hipErrorInvalidValue
    hipError_t hipModuleGetTexRef(textureReference ** texRef,hipModule_t hmod,const char * name) nogil


    # @brief builds module from code object which resides in host memory. Image is pointer to that
    # location.
    # @param [in] image
    # @param [out] module
    # @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
    hipError_t hipModuleLoadData(hipModule_t* module,const void * image) nogil


    # @brief builds module from code object which resides in host memory. Image is pointer to that
    # location. Options are not used. hipModuleLoadData is called.
    # @param [in] image
    # @param [out] module
    # @param [in] number of options
    # @param [in] options for JIT
    # @param [in] option values for JIT
    # @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
    hipError_t hipModuleLoadDataEx(hipModule_t* module,const void * image,unsigned int numOptions,hipJitOption * options,void ** optionValues) nogil


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
    hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,hipStream_t stream,void ** kernelParams,void ** extra) nogil


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
    hipError_t hipLaunchCooperativeKernel(const void * f,dim3 gridDim,dim3 blockDimX,void ** kernelParams,unsigned int sharedMemBytes,hipStream_t stream) nogil


    # @brief Launches kernels on multiple devices where thread blocks can cooperate and
    # synchronize as they execute.
    # @param [in] launchParamsList         List of launch parameters, one per device.
    # @param [in] numDevices               Size of the launchParamsList array.
    # @param [in] flags                    Flags to control launch behavior.
    # @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
    hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams_t * launchParamsList,int numDevices,unsigned int flags) nogil


    # @brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
    # on respective streams before enqueuing any other work on the specified streams from any other threads
    # @param [in] hipLaunchParams          List of launch parameters, one per device.
    # @param [in] numDevices               Size of the launchParamsList array.
    # @param [in] flags                    Flags to control launch behavior.
    # @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams_t * launchParamsList,int numDevices,unsigned int flags) nogil


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
    hipError_t hipModuleOccupancyMaxPotentialBlockSize(int * gridSize,int * blockSize,hipFunction_t f,int dynSharedMemPerBlk,int blockSizeLimit) nogil


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
    hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int * gridSize,int * blockSize,hipFunction_t f,int dynSharedMemPerBlk,int blockSizeLimit,unsigned int flags) nogil


    # @brief Returns occupancy for a device function.
    # @param [out] numBlocks        Returned occupancy
    # @param [in]  func             Kernel function (hipFunction) for which occupancy is calulated
    # @param [in]  blockSize        Block size the kernel is intended to be launched with
    # @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,hipFunction_t f,int blockSize,int dynSharedMemPerBlk) nogil


    # @brief Returns occupancy for a device function.
    # @param [out] numBlocks        Returned occupancy
    # @param [in]  f                Kernel function(hipFunction_t) for which occupancy is calulated
    # @param [in]  blockSize        Block size the kernel is intended to be launched with
    # @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    # @param [in]  flags            Extra flags for occupancy calculation (only default supported)
    hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,hipFunction_t f,int blockSize,int dynSharedMemPerBlk,unsigned int flags) nogil


    # @brief Returns occupancy for a device function.
    # @param [out] numBlocks        Returned occupancy
    # @param [in]  func             Kernel function for which occupancy is calulated
    # @param [in]  blockSize        Block size the kernel is intended to be launched with
    # @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,const void * f,int blockSize,int dynSharedMemPerBlk) nogil


    # @brief Returns occupancy for a device function.
    # @param [out] numBlocks        Returned occupancy
    # @param [in]  f                Kernel function for which occupancy is calulated
    # @param [in]  blockSize        Block size the kernel is intended to be launched with
    # @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    # @param [in]  flags            Extra flags for occupancy calculation (currently ignored)
    hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,const void * f,int blockSize,int dynSharedMemPerBlk,unsigned int flags) nogil


    # @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
    # @param [out] gridSize           minimum grid size for maximum potential occupancy
    # @param [out] blockSize          block size for maximum potential occupancy
    # @param [in]  f                  kernel function for which occupancy is calulated
    # @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    # @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
    # Please note, HIP does not support kernel launch with total work items defined in dimension with
    # size gridDim x blockDim >= 2^32.
    # @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
    hipError_t hipOccupancyMaxPotentialBlockSize(int * gridSize,int * blockSize,const void * f,int dynSharedMemPerBlk,int blockSizeLimit) nogil


    # @brief Start recording of profiling information
    # When using this API, start the profiler with profiling disabled.  (--startdisabled)
    # @warning : hipProfilerStart API is under development.
    hipError_t hipProfilerStart() nogil


    # @brief Stop recording of profiling information.
    # When using this API, start the profiler with profiling disabled.  (--startdisabled)
    # @warning : hipProfilerStop API is under development.
    hipError_t hipProfilerStop() nogil


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
    hipError_t hipConfigureCall(dim3 gridDim,dim3 blockDim,int sharedMem,hipStream_t stream) nogil


    # @brief Set a kernel argument.
    # @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    # @param [in] arg    Pointer the argument in host memory.
    # @param [in] size   Size of the argument.
    # @param [in] offset Offset of the argument on the argument stack.
    hipError_t hipSetupArgument(const void * arg,int size,int offset) nogil


    # @brief Launch a kernel.
    # @param [in] func Kernel to launch.
    # @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    hipError_t hipLaunchByPtr(const void * func) nogil


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
    hipError_t hipLaunchKernel(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,int sharedMemBytes,hipStream_t stream) nogil


    # @brief Enqueues a host function call in a stream.
    # @param [in] stream - stream to enqueue work to.
    # @param [in] fn - function to call once operations enqueued preceeding are complete.
    # @param [in] userData - User-specified data to be passed to the function.
    # @returns #hipSuccess, #hipErrorInvalidResourceHandle, #hipErrorInvalidValue,
    # #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipLaunchHostFunc(hipStream_t stream,hipHostFn_t fn,void * userData) nogil


    # Copies memory for 2D arrays.
    # @param pCopy           - Parameters for the memory copy
    # @returns #hipSuccess, #hipErrorInvalidValue
    hipError_t hipDrvMemcpy2DUnaligned(hip_Memcpy2D * pCopy) nogil


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
    hipError_t hipExtLaunchKernel(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,int sharedMemBytes,hipStream_t stream,hipEvent_t startEvent,hipEvent_t stopEvent,int flags) nogil


    # @brief  Binds a mipmapped array to a texture.
    # @param [in] tex  pointer to the texture reference to bind
    # @param [in] mipmappedArray  memory mipmapped array on the device
    # @param [in] desc  opointer to the channel format
    # @returns hipSuccess, hipErrorInvalidValue
    hipError_t hipBindTextureToMipmappedArray(textureReference * tex,hipMipmappedArray_const_t mipmappedArray,hipChannelFormatDesc * desc) nogil


    # @brief Creates a texture object.
    # @param [out] pTexObject  pointer to the texture object to create
    # @param [in] pResDesc  pointer to resource descriptor
    # @param [in] pTexDesc  pointer to texture descriptor
    # @param [in] pResViewDesc  pointer to resource view descriptor
    # @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported, hipErrorOutOfMemory
    # @note 3D liner filter isn't supported on GFX90A boards, on which the API @p hipCreateTextureObject will
    # return hipErrorNotSupported.
    hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject,hipResourceDesc * pResDesc,hipTextureDesc * pTexDesc,hipResourceViewDesc * pResViewDesc) nogil


    # @brief Destroys a texture object.
    # @param [in] textureObject  texture object to destroy
    # @returns hipSuccess, hipErrorInvalidValue
    hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) nogil


    # @brief Gets the channel descriptor in an array.
    # @param [in] desc  pointer to channel format descriptor
    # @param [out] array  memory array on the device
    # @returns hipSuccess, hipErrorInvalidValue
    hipError_t hipGetChannelDesc(hipChannelFormatDesc * desc,hipArray_const_t array) nogil


    # @brief Gets resource descriptor for the texture object.
    # @param [out] pResDesc  pointer to resource descriptor
    # @param [in] textureObject  texture object
    # @returns hipSuccess, hipErrorInvalidValue
    hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc * pResDesc,hipTextureObject_t textureObject) nogil


    # @brief Gets resource view descriptor for the texture object.
    # @param [out] pResViewDesc  pointer to resource view descriptor
    # @param [in] textureObject  texture object
    # @returns hipSuccess, hipErrorInvalidValue
    hipError_t hipGetTextureObjectResourceViewDesc(hipResourceViewDesc * pResViewDesc,hipTextureObject_t textureObject) nogil


    # @brief Gets texture descriptor for the texture object.
    # @param [out] pTexDesc  pointer to texture descriptor
    # @param [in] textureObject  texture object
    # @returns hipSuccess, hipErrorInvalidValue
    hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc * pTexDesc,hipTextureObject_t textureObject) nogil


    # @brief Creates a texture object.
    # @param [out] pTexObject  pointer to texture object to create
    # @param [in] pResDesc  pointer to resource descriptor
    # @param [in] pTexDesc  pointer to texture descriptor
    # @param [in] pResViewDesc  pointer to resource view descriptor
    # @returns hipSuccess, hipErrorInvalidValue
    hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject,HIP_RESOURCE_DESC_st * pResDesc,HIP_TEXTURE_DESC_st * pTexDesc,HIP_RESOURCE_VIEW_DESC_st * pResViewDesc) nogil


    # @brief Destroys a texture object.
    # @param [in] texObject  texture object to destroy
    # @returns hipSuccess, hipErrorInvalidValue
    hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) nogil


    # @brief Gets resource descriptor of a texture object.
    # @param [out] pResDesc  pointer to resource descriptor
    # @param [in] texObject  texture object
    # @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC_st * pResDesc,hipTextureObject_t texObject) nogil


    # @brief Gets resource view descriptor of a texture object.
    # @param [out] pResViewDesc  pointer to resource view descriptor
    # @param [in] texObject  texture object
    # @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC_st * pResViewDesc,hipTextureObject_t texObject) nogil


    # @brief Gets texture descriptor of a texture object.
    # @param [out] pTexDesc  pointer to texture descriptor
    # @param [in] texObject  texture object
    # @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC_st * pTexDesc,hipTextureObject_t texObject) nogil


    # @addtogroup TextureD Texture Management [Deprecated]
    # @{
    # @ingroup Texture
    # This section describes the deprecated texture management functions of HIP runtime API.
    # @brief Gets the texture reference related with the symbol.
    # @param [out] texref  texture reference
    # @param [in] symbol  pointer to the symbol related with the texture for the reference
    # @returns hipSuccess, hipErrorInvalidValue
    hipError_t hipGetTextureReference(textureReference ** texref,const void * symbol) nogil



    hipError_t hipTexRefSetAddressMode(textureReference * texRef,int dim,hipTextureAddressMode am) nogil



    hipError_t hipTexRefSetArray(textureReference * tex,hipArray_const_t array,unsigned int flags) nogil



    hipError_t hipTexRefSetFilterMode(textureReference * texRef,hipTextureFilterMode fm) nogil



    hipError_t hipTexRefSetFlags(textureReference * texRef,unsigned int Flags) nogil



    hipError_t hipTexRefSetFormat(textureReference * texRef,hipArray_Format fmt,int NumPackedComponents) nogil



    hipError_t hipBindTexture(int * offset,textureReference * tex,const void * devPtr,hipChannelFormatDesc * desc,int size) nogil



    hipError_t hipBindTexture2D(int * offset,textureReference * tex,const void * devPtr,hipChannelFormatDesc * desc,int width,int height,int pitch) nogil



    hipError_t hipBindTextureToArray(textureReference * tex,hipArray_const_t array,hipChannelFormatDesc * desc) nogil



    hipError_t hipGetTextureAlignmentOffset(int * offset,textureReference * texref) nogil



    hipError_t hipUnbindTexture(textureReference * tex) nogil



    hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr,textureReference * texRef) nogil



    hipError_t hipTexRefGetAddressMode(hipTextureAddressMode * pam,textureReference * texRef,int dim) nogil



    hipError_t hipTexRefGetFilterMode(hipTextureFilterMode * pfm,textureReference * texRef) nogil



    hipError_t hipTexRefGetFlags(unsigned int * pFlags,textureReference * texRef) nogil



    hipError_t hipTexRefGetFormat(hipArray_Format * pFormat,int * pNumChannels,textureReference * texRef) nogil



    hipError_t hipTexRefGetMaxAnisotropy(int * pmaxAnsio,textureReference * texRef) nogil



    hipError_t hipTexRefGetMipmapFilterMode(hipTextureFilterMode * pfm,textureReference * texRef) nogil



    hipError_t hipTexRefGetMipmapLevelBias(float * pbias,textureReference * texRef) nogil



    hipError_t hipTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp,float * pmaxMipmapLevelClamp,textureReference * texRef) nogil



    hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray,textureReference * texRef) nogil



    hipError_t hipTexRefSetAddress(int * ByteOffset,textureReference * texRef,hipDeviceptr_t dptr,int bytes) nogil



    hipError_t hipTexRefSetAddress2D(textureReference * texRef,HIP_ARRAY_DESCRIPTOR * desc,hipDeviceptr_t dptr,int Pitch) nogil



    hipError_t hipTexRefSetMaxAnisotropy(textureReference * texRef,unsigned int maxAniso) nogil



    hipError_t hipTexRefSetBorderColor(textureReference * texRef,float * pBorderColor) nogil



    hipError_t hipTexRefSetMipmapFilterMode(textureReference * texRef,hipTextureFilterMode fm) nogil



    hipError_t hipTexRefSetMipmapLevelBias(textureReference * texRef,float bias) nogil



    hipError_t hipTexRefSetMipmapLevelClamp(textureReference * texRef,float minMipMapLevelClamp,float maxMipMapLevelClamp) nogil



    hipError_t hipTexRefSetMipmappedArray(textureReference * texRef,hipMipmappedArray * mipmappedArray,unsigned int Flags) nogil


    # @addtogroup TextureU Texture Management [Not supported]
    # @{
    # @ingroup Texture
    # This section describes the texture management functions currently unsupported in HIP runtime.
    hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,HIP_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc,unsigned int numMipmapLevels) nogil



    hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) nogil



    hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray,hipMipmappedArray_t hMipMappedArray,unsigned int level) nogil


    # @defgroup Callback Callback Activity APIs
    # @{
    # This section describes the callback/Activity of HIP runtime API.
    const char * hipApiName(uint32_t id) nogil



    const char * hipKernelNameRef(hipFunction_t f) nogil



    const char * hipKernelNameRefByPtr(const void * hostFunction,hipStream_t stream) nogil



    int hipGetStreamDeviceId(hipStream_t stream) nogil


    # @brief Begins graph capture on a stream.
    # @param [in] stream - Stream to initiate capture.
    # @param [in] mode - Controls the interaction of this capture sequence with other API calls that
    # are not safe.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipStreamBeginCapture(hipStream_t stream,hipStreamCaptureMode mode) nogil


    # @brief Ends capture on a stream, returning the captured graph.
    # @param [in] stream - Stream to end capture.
    # @param [out] pGraph - returns the graph captured.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipStreamEndCapture(hipStream_t stream,hipGraph_t* pGraph) nogil


    # @brief Get capture status of a stream.
    # @param [in] stream - Stream under capture.
    # @param [out] pCaptureStatus - returns current status of the capture.
    # @param [out] pId - unique ID of the capture.
    # @returns #hipSuccess, #hipErrorStreamCaptureImplicit
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipStreamGetCaptureInfo(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus,unsigned long long * pId) nogil


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
    hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream,hipStreamCaptureStatus * captureStatus_out,unsigned long long * id_out,hipGraph_t* graph_out,hipGraphNode_t ** dependencies_out,int * numDependencies_out) nogil


    # @brief Get stream's capture state
    # @param [in] stream - Stream under capture.
    # @param [out] pCaptureStatus - returns current status of the capture.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipStreamIsCapturing(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus) nogil


    # @brief Update the set of dependencies in a capturing stream
    # @param [in] stream - Stream under capture.
    # @param [in] dependencies - pointer to an array of nodes to Add/Replace.
    # @param [in] numDependencies - size of the array in dependencies.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorIllegalState
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream,hipGraphNode_t* dependencies,int numDependencies,unsigned int flags) nogil


    # @brief Swaps the stream capture mode of a thread.
    # @param [in] mode - Pointer to mode value to swap with the current mode
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode * mode) nogil


    # @brief Creates a graph
    # @param [out] pGraph - pointer to graph to create.
    # @param [in] flags - flags for graph creation, must be 0.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphCreate(hipGraph_t* pGraph,unsigned int flags) nogil


    # @brief Destroys a graph
    # @param [in] graph - instance of graph to destroy.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphDestroy(hipGraph_t graph) nogil


    # @brief Adds dependency edges to a graph.
    # @param [in] graph - instance of the graph to add dependencies.
    # @param [in] from - pointer to the graph nodes with dependenties to add from.
    # @param [in] to - pointer to the graph nodes to add dependenties to.
    # @param [in] numDependencies - the number of dependencies to add.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphAddDependencies(hipGraph_t graph,hipGraphNode_t * from_,hipGraphNode_t * to,int numDependencies) nogil


    # @brief Removes dependency edges from a graph.
    # @param [in] graph - instance of the graph to remove dependencies.
    # @param [in] from - Array of nodes that provide the dependencies.
    # @param [in] to - Array of dependent nodes.
    # @param [in] numDependencies - the number of dependencies to remove.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphRemoveDependencies(hipGraph_t graph,hipGraphNode_t * from_,hipGraphNode_t * to,int numDependencies) nogil


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
    hipError_t hipGraphGetEdges(hipGraph_t graph,hipGraphNode_t* from_,hipGraphNode_t* to,int * numEdges) nogil


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
    hipError_t hipGraphGetNodes(hipGraph_t graph,hipGraphNode_t* nodes,int * numNodes) nogil


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
    hipError_t hipGraphGetRootNodes(hipGraph_t graph,hipGraphNode_t* pRootNodes,int * pNumRootNodes) nogil


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
    hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node,hipGraphNode_t* pDependencies,int * pNumDependencies) nogil


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
    hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node,hipGraphNode_t* pDependentNodes,int * pNumDependentNodes) nogil


    # @brief Returns a node's type.
    # @param [in] node - instance of the graph to add dependencies.
    # @param [out] pType - pointer to the return the type
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphNodeGetType(hipGraphNode_t node,hipGraphNodeType * pType) nogil


    # @brief Remove a node from the graph.
    # @param [in] node - graph node to remove
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphDestroyNode(hipGraphNode_t node) nogil


    # @brief Clones a graph.
    # @param [out] pGraphClone - Returns newly created cloned graph.
    # @param [in] originalGraph - original graph to clone from.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphClone(hipGraph_t* pGraphClone,hipGraph_t originalGraph) nogil


    # @brief Finds a cloned version of a node.
    # @param [out] pNode - Returns the cloned node.
    # @param [in] originalNode - original node handle.
    # @param [in] clonedGraph - Cloned graph to query.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode,hipGraphNode_t originalNode,hipGraph_t clonedGraph) nogil


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
    hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec,hipGraph_t graph,hipGraphNode_t* pErrorNode,char * pLogBuffer,int bufferSize) nogil


    # @brief Creates an executable graph from a graph.
    # @param [out] pGraphExec - pointer to instantiated executable graph that is created.
    # @param [in] graph - instance of graph to instantiate.
    # @param [in] flags - Flags to control instantiation.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec,hipGraph_t graph,unsigned long long flags) nogil


    # @brief launches an executable graph in a stream
    # @param [in] graphExec - instance of executable graph to launch.
    # @param [in] stream - instance of stream in which to launch executable graph.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphLaunch(hipGraphExec_t graphExec,hipStream_t stream) nogil


    # @brief uploads an executable graph in a stream
    # @param [in] graphExec - instance of executable graph to launch.
    # @param [in] stream - instance of stream in which to launch executable graph.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphUpload(hipGraphExec_t graphExec,hipStream_t stream) nogil


    # @brief Destroys an executable graph
    # @param [in] pGraphExec - instance of executable graph to destry.
    # @returns #hipSuccess.
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) nogil


    # @brief Check whether an executable graph can be updated with a graph and perform the update if  *
    # possible.
    # @param [in] hGraphExec - instance of executable graph to update.
    # @param [in] hGraph - graph that contains the updated parameters.
    # @param [in] hErrorNode_out -  node which caused the permissibility check to forbid the update.
    # @param [in] updateResult_out - Whether the graph update was permitted.
    # @returns #hipSuccess, #hipErrorGraphExecUpdateFailure
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec,hipGraph_t hGraph,hipGraphNode_t* hErrorNode_out,hipGraphExecUpdateResult * updateResult_out) nogil


    # @brief Creates a kernel execution node and adds it to a graph.
    # @param [out] pGraphNode - pointer to graph node to create.
    # @param [in] graph - instance of graph to add the created node.
    # @param [in] pDependencies - pointer to the dependencies on the kernel execution node.
    # @param [in] numDependencies - the number of the dependencies.
    # @param [in] pNodeParams - pointer to the parameters to the kernel execution node on the GPU.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDeviceFunction
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipKernelNodeParams * pNodeParams) nogil


    # @brief Gets kernel node's parameters.
    # @param [in] node - instance of the node to get parameters from.
    # @param [out] pNodeParams - pointer to the parameters
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil


    # @brief Sets a kernel node's parameters.
    # @param [in] node - instance of the node to set parameters to.
    # @param [in] pNodeParams - const pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil


    # @brief Sets the parameters for a kernel node in the given graphExec.
    # @param [in] hGraphExec - instance of the executable graph with the node.
    # @param [in] node - instance of the node to set parameters to.
    # @param [in] pNodeParams - const pointer to the kernel node parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil


    # @brief Creates a memcpy node and adds it to a graph.
    # @param [out] pGraphNode - pointer to graph node to create.
    # @param [in] graph - instance of graph to add the created node.
    # @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
    # @param [in] numDependencies - the number of the dependencies.
    # @param [in] pCopyParams - const pointer to the parameters for the memory copy.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipMemcpy3DParms * pCopyParams) nogil


    # @brief Gets a memcpy node's parameters.
    # @param [in] node - instance of the node to get parameters from.
    # @param [out] pNodeParams - pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil


    # @brief Sets a memcpy node's parameters.
    # @param [in] node - instance of the node to set parameters to.
    # @param [in] pNodeParams - const pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil


    # @brief Sets a node attribute.
    # @param [in] hNode - instance of the node to set parameters to.
    # @param [in] attr - the attribute node is set to.
    # @param [in] value - const pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode,hipKernelNodeAttrID attr,hipKernelNodeAttrValue * value) nogil


    # @brief Gets a node attribute.
    # @param [in] hNode - instance of the node to set parameters to.
    # @param [in] attr - the attribute node is set to.
    # @param [in] value - const pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode,hipKernelNodeAttrID attr,hipKernelNodeAttrValue * value) nogil


    # @brief Sets the parameters for a memcpy node in the given graphExec.
    # @param [in] hGraphExec - instance of the executable graph with the node.
    # @param [in] node - instance of the node to set parameters to.
    # @param [in] pNodeParams - const pointer to the kernel node parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil


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
    hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,void * dst,const void * src,int count,hipMemcpyKind kind) nogil


    # @brief Sets a memcpy node's parameters to perform a 1-dimensional copy.
    # @param [in] node - instance of the node to set parameters to.
    # @param [in] dst - pointer to memory address to the destination.
    # @param [in] src - pointer to memory address to the source.
    # @param [in] count - the size of the memory to copy.
    # @param [in] kind - the type of memory copy.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node,void * dst,const void * src,int count,hipMemcpyKind kind) nogil


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
    hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec,hipGraphNode_t node,void * dst,const void * src,int count,hipMemcpyKind kind) nogil


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
    hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,void * dst,const void * symbol,int count,int offset,hipMemcpyKind kind) nogil


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
    hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node,void * dst,const void * symbol,int count,int offset,hipMemcpyKind kind) nogil


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
    hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec,hipGraphNode_t node,void * dst,const void * symbol,int count,int offset,hipMemcpyKind kind) nogil


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
    hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,const void * symbol,const void * src,int count,int offset,hipMemcpyKind kind) nogil


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
    hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node,const void * symbol,const void * src,int count,int offset,hipMemcpyKind kind) nogil


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
    hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec,hipGraphNode_t node,const void * symbol,const void * src,int count,int offset,hipMemcpyKind kind) nogil


    # @brief Creates a memset node and adds it to a graph.
    # @param [out] pGraphNode - pointer to the graph node to create.
    # @param [in] graph - instance of the graph to add the created node.
    # @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
    # @param [in] numDependencies - the number of the dependencies.
    # @param [in] pMemsetParams - const pointer to the parameters for the memory set.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipMemsetParams * pMemsetParams) nogil


    # @brief Gets a memset node's parameters.
    # @param [in] node - instane of the node to get parameters from.
    # @param [out] pNodeParams - pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil


    # @brief Sets a memset node's parameters.
    # @param [in] node - instance of the node to set parameters to.
    # @param [in] pNodeParams - pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil


    # @brief Sets the parameters for a memset node in the given graphExec.
    # @param [in] hGraphExec - instance of the executable graph with the node.
    # @param [in] node - instance of the node to set parameters to.
    # @param [in] pNodeParams - pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil


    # @brief Creates a host execution node and adds it to a graph.
    # @param [out] pGraphNode - pointer to the graph node to create.
    # @param [in] graph - instance of the graph to add the created node.
    # @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
    # @param [in] numDependencies - the number of the dependencies.
    # @param [in] pNodeParams -pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipHostNodeParams * pNodeParams) nogil


    # @brief Returns a host node's parameters.
    # @param [in] node - instane of the node to get parameters from.
    # @param [out] pNodeParams - pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil


    # @brief Sets a host node's parameters.
    # @param [in] node - instance of the node to set parameters to.
    # @param [in] pNodeParams - pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil


    # @brief Sets the parameters for a host node in the given graphExec.
    # @param [in] hGraphExec - instance of the executable graph with the node.
    # @param [in] node - instance of the node to set parameters to.
    # @param [in] pNodeParams - pointer to the parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil


    # @brief Creates a child graph node and adds it to a graph.
    # @param [out] pGraphNode - pointer to the graph node to create.
    # @param [in] graph - instance of the graph to add the created node.
    # @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
    # @param [in] numDependencies - the number of the dependencies.
    # @param [in] childGraph - the graph to clone into this node
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipGraph_t childGraph) nogil


    # @brief Gets a handle to the embedded graph of a child graph node.
    # @param [in] node - instane of the node to get child graph.
    # @param [out] pGraph - pointer to get the graph.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node,hipGraph_t* pGraph) nogil


    # @brief Updates node parameters in the child graph node in the given graphExec.
    # @param [in] hGraphExec - instance of the executable graph with the node.
    # @param [in] node - node from the graph which was used to instantiate graphExec.
    # @param [in] childGraph - child graph with updated parameters.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipGraph_t childGraph) nogil


    # @brief Creates an empty node and adds it to a graph.
    # @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    # @param [in] graph - instane of the graph the node is add to.
    # @param [in] pDependencies - const pointer to the node dependenties.
    # @param [in] numDependencies - the number of dependencies.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies) nogil


    # @brief Creates an event record node and adds it to a graph.
    # @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    # @param [in] graph - instane of the graph the node to be added.
    # @param [in] pDependencies - const pointer to the node dependenties.
    # @param [in] numDependencies - the number of dependencies.
    # @param [in] event - Event for the node.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipEvent_t event) nogil


    # @brief Returns the event associated with an event record node.
    # @param [in] node -  instane of the node to get event from.
    # @param [out] event_out - Pointer to return the event.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node,hipEvent_t* event_out) nogil


    # @brief Sets an event record node's event.
    # @param [in] node - instane of the node to set event to.
    # @param [in] event - pointer to the event.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node,hipEvent_t event) nogil


    # @brief Sets the event for an event record node in the given graphExec.
    # @param [in] hGraphExec - instance of the executable graph with the node.
    # @param [in] hNode - node from the graph which was used to instantiate graphExec.
    # @param [in] event - pointer to the event.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec,hipGraphNode_t hNode,hipEvent_t event) nogil


    # @brief Creates an event wait node and adds it to a graph.
    # @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    # @param [in] graph - instane of the graph the node to be added.
    # @param [in] pDependencies - const pointer to the node dependenties.
    # @param [in] numDependencies - the number of dependencies.
    # @param [in] event - Event for the node.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode,hipGraph_t graph,hipGraphNode_t * pDependencies,int numDependencies,hipEvent_t event) nogil


    # @brief Returns the event associated with an event wait node.
    # @param [in] node -  instane of the node to get event from.
    # @param [out] event_out - Pointer to return the event.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node,hipEvent_t* event_out) nogil


    # @brief Sets an event wait node's event.
    # @param [in] node - instane of the node to set event to.
    # @param [in] event - pointer to the event.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node,hipEvent_t event) nogil


    # @brief Sets the event for an event record node in the given graphExec.
    # @param [in] hGraphExec - instance of the executable graph with the node.
    # @param [in] hNode - node from the graph which was used to instantiate graphExec.
    # @param [in] event - pointer to the event.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec,hipGraphNode_t hNode,hipEvent_t event) nogil


    # @brief Get the mem attribute for graphs.
    # @param [in] device - device the attr is get for.
    # @param [in] attr - attr to get.
    # @param [out] value - value for specific attr.
    # @returns #hipSuccess, #hipErrorInvalidDevice
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipDeviceGetGraphMemAttribute(int device,hipGraphMemAttributeType attr,void * value) nogil


    # @brief Set the mem attribute for graphs.
    # @param [in] device - device the attr is set for.
    # @param [in] attr - attr to set.
    # @param [in] value - value for specific attr.
    # @returns #hipSuccess, #hipErrorInvalidDevice
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipDeviceSetGraphMemAttribute(int device,hipGraphMemAttributeType attr,void * value) nogil


    # @brief Free unused memory on specific device used for graph back to OS.
    # @param [in] device - device the memory is used for graphs
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipDeviceGraphMemTrim(int device) nogil


    # @brief Create an instance of userObject to manage lifetime of a resource.
    # @param [out] object_out - pointer to instace of userobj.
    # @param [in] ptr - pointer to pass to destroy function.
    # @param [in] destroy - destroy callback to remove resource.
    # @param [in] initialRefcount - reference to resource.
    # @param [in] flags - flags passed to API.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipUserObjectCreate(hipUserObject_t* object_out,void * ptr,hipHostFn_t destroy,unsigned int initialRefcount,unsigned int flags) nogil


    # @brief Release number of references to resource.
    # @param [in] object - pointer to instace of userobj.
    # @param [in] count - reference to resource to be retained.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipUserObjectRelease(hipUserObject_t object,unsigned int count) nogil


    # @brief Retain number of references to resource.
    # @param [in] object - pointer to instace of userobj.
    # @param [in] count - reference to resource to be retained.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipUserObjectRetain(hipUserObject_t object,unsigned int count) nogil


    # @brief Retain user object for graphs.
    # @param [in] graph - pointer to graph to retain the user object for.
    # @param [in] object - pointer to instace of userobj.
    # @param [in] count - reference to resource to be retained.
    # @param [in] flags - flags passed to API.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphRetainUserObject(hipGraph_t graph,hipUserObject_t object,unsigned int count,unsigned int flags) nogil


    # @brief Release user object from graphs.
    # @param [in] graph - pointer to graph to retain the user object for.
    # @param [in] object - pointer to instace of userobj.
    # @param [in] count - reference to resource to be retained.
    # @returns #hipSuccess, #hipErrorInvalidValue
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipGraphReleaseUserObject(hipGraph_t graph,hipUserObject_t object,unsigned int count) nogil


    # @brief Frees an address range reservation made via hipMemAddressReserve
    # @param [in] devPtr - starting address of the range.
    # @param [in] size - size of the range.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemAddressFree(void * devPtr,int size) nogil


    # @brief Reserves an address range
    # @param [out] ptr - starting address of the reserved range.
    # @param [in] size - size of the reservation.
    # @param [in] alignment - alignment of the address.
    # @param [in] addr - requested starting address of the range.
    # @param [in] flags - currently unused, must be zero.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemAddressReserve(void ** ptr,int size,int alignment,void * addr,unsigned long long flags) nogil


    # @brief Creates a memory allocation described by the properties and size
    # @param [out] handle - value of the returned handle.
    # @param [in] size - size of the allocation.
    # @param [in] prop - properties of the allocation.
    # @param [in] flags - currently unused, must be zero.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle,int size,hipMemAllocationProp * prop,unsigned long long flags) nogil


    # @brief Exports an allocation to a requested shareable handle type.
    # @param [out] shareableHandle - value of the returned handle.
    # @param [in] handle - handle to share.
    # @param [in] handleType - type of the shareable handle.
    # @param [in] flags - currently unused, must be zero.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemExportToShareableHandle(void * shareableHandle,hipMemGenericAllocationHandle_t handle,hipMemAllocationHandleType handleType,unsigned long long flags) nogil


    # @brief Get the access flags set for the given location and ptr.
    # @param [out] flags - flags for this location.
    # @param [in] location - target location.
    # @param [in] ptr - address to check the access flags.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemGetAccess(unsigned long long * flags,hipMemLocation * location,void * ptr) nogil


    # @brief Calculates either the minimal or recommended granularity.
    # @param [out] granularity - returned granularity.
    # @param [in] prop - location properties.
    # @param [in] option - determines which granularity to return.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemGetAllocationGranularity(int * granularity,hipMemAllocationProp * prop,hipMemAllocationGranularity_flags option) nogil


    # @brief Retrieve the property structure of the given handle.
    # @param [out] prop - properties of the given handle.
    # @param [in] handle - handle to perform the query on.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp * prop,hipMemGenericAllocationHandle_t handle) nogil


    # @brief Imports an allocation from a requested shareable handle type.
    # @param [out] handle - returned value.
    # @param [in] osHandle - shareable handle representing the memory allocation.
    # @param [in] shHandleType - handle type.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle,void * osHandle,hipMemAllocationHandleType shHandleType) nogil


    # @brief Maps an allocation handle to a reserved virtual address range.
    # @param [in] ptr - address where the memory will be mapped.
    # @param [in] size - size of the mapping.
    # @param [in] offset - offset into the memory, currently must be zero.
    # @param [in] handle - memory allocation to be mapped.
    # @param [in] flags - currently unused, must be zero.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemMap(void * ptr,int size,int offset,hipMemGenericAllocationHandle_t handle,unsigned long long flags) nogil


    # @brief Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.
    # @param [in] mapInfoList - list of hipArrayMapInfo.
    # @param [in] count - number of hipArrayMapInfo in mapInfoList.
    # @param [in] stream - stream identifier for the stream to use for map or unmap operations.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemMapArrayAsync(hipArrayMapInfo * mapInfoList,unsigned int count,hipStream_t stream) nogil


    # @brief Release a memory handle representing a memory allocation which was previously allocated through hipMemCreate.
    # @param [in] handle - handle of the memory allocation.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) nogil


    # @brief Returns the allocation handle of the backing memory allocation given the address.
    # @param [out] handle - handle representing addr.
    # @param [in] addr - address to look up.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle,void * addr) nogil


    # @brief Set the access flags for each location specified in desc for the given virtual address range.
    # @param [in] ptr - starting address of the virtual address range.
    # @param [in] size - size of the range.
    # @param [in] desc - array of hipMemAccessDesc.
    # @param [in] count - number of hipMemAccessDesc in desc.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemSetAccess(void * ptr,int size,hipMemAccessDesc * desc,int count) nogil


    # @brief Unmap memory allocation of a given address range.
    # @param [in] ptr - starting address of the range to unmap.
    # @param [in] size - size of the virtual address range.
    # @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    # @warning : This API is marked as beta, meaning, while this is feature complete,
    # it is still open to changes and may have outstanding issues.
    hipError_t hipMemUnmap(void * ptr,int size) nogil


    ctypedef unsigned int GLuint

    ctypedef unsigned int GLenum


    hipError_t hipGLGetDevices(unsigned int * pHipDeviceCount,int * pHipDevices,unsigned int hipDeviceCount,hipGLDeviceList deviceList) nogil



    hipError_t hipGraphicsGLRegisterBuffer(_hipGraphicsResource ** resource,GLuint buffer,unsigned int flags) nogil



    hipError_t hipGraphicsGLRegisterImage(_hipGraphicsResource ** resource,GLuint image,GLenum target,unsigned int flags) nogil



    hipError_t hipGraphicsMapResources(int count,hipGraphicsResource_t* resources,hipStream_t stream) nogil



    hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array,hipGraphicsResource_t resource,unsigned int arrayIndex,unsigned int mipLevel) nogil



    hipError_t hipGraphicsResourceGetMappedPointer(void ** devPtr,int * size,hipGraphicsResource_t resource) nogil



    hipError_t hipGraphicsUnmapResources(int count,hipGraphicsResource_t* resources,hipStream_t stream) nogil



    hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) nogil



    hipError_t hipMemcpy_spt(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind) nogil



    hipError_t hipMemcpyToSymbol_spt(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind) nogil



    hipError_t hipMemcpyFromSymbol_spt(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind) nogil



    hipError_t hipMemcpy2D_spt(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil



    hipError_t hipMemcpy2DFromArray_spt(void * dst,int dpitch,hipArray_const_t src,int wOffset,int hOffset,int width,int height,hipMemcpyKind kind) nogil



    hipError_t hipMemcpy3D_spt(hipMemcpy3DParms * p) nogil



    hipError_t hipMemset_spt(void * dst,int value,int sizeBytes) nogil



    hipError_t hipMemsetAsync_spt(void * dst,int value,int sizeBytes,hipStream_t stream) nogil



    hipError_t hipMemset2D_spt(void * dst,int pitch,int value,int width,int height) nogil



    hipError_t hipMemset2DAsync_spt(void * dst,int pitch,int value,int width,int height,hipStream_t stream) nogil



    hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent,hipStream_t stream) nogil



    hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent) nogil



    hipError_t hipMemcpyAsync_spt(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil



    hipError_t hipMemcpy3DAsync_spt(hipMemcpy3DParms * p,hipStream_t stream) nogil



    hipError_t hipMemcpy2DAsync_spt(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil



    hipError_t hipMemcpyFromSymbolAsync_spt(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil



    hipError_t hipMemcpyToSymbolAsync_spt(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil



    hipError_t hipMemcpyFromArray_spt(void * dst,hipArray_const_t src,int wOffsetSrc,int hOffset,int count,hipMemcpyKind kind) nogil



    hipError_t hipMemcpy2DToArray_spt(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil



    hipError_t hipMemcpy2DFromArrayAsync_spt(void * dst,int dpitch,hipArray_const_t src,int wOffsetSrc,int hOffsetSrc,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil



    hipError_t hipMemcpy2DToArrayAsync_spt(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil



    hipError_t hipStreamQuery_spt(hipStream_t stream) nogil



    hipError_t hipStreamSynchronize_spt(hipStream_t stream) nogil



    hipError_t hipStreamGetPriority_spt(hipStream_t stream,int * priority) nogil



    hipError_t hipStreamWaitEvent_spt(hipStream_t stream,hipEvent_t event,unsigned int flags) nogil



    hipError_t hipStreamGetFlags_spt(hipStream_t stream,unsigned int * flags) nogil



    hipError_t hipStreamAddCallback_spt(hipStream_t stream,hipStreamCallback_t callback,void * userData,unsigned int flags) nogil



    hipError_t hipEventRecord_spt(hipEvent_t event,hipStream_t stream) nogil



    hipError_t hipLaunchCooperativeKernel_spt(const void * f,dim3 gridDim,dim3 blockDim,void ** kernelParams,uint32_t sharedMemBytes,hipStream_t hStream) nogil



    hipError_t hipLaunchKernel_spt(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,int sharedMemBytes,hipStream_t stream) nogil



    hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec,hipStream_t stream) nogil



    hipError_t hipStreamBeginCapture_spt(hipStream_t stream,hipStreamCaptureMode mode) nogil



    hipError_t hipStreamEndCapture_spt(hipStream_t stream,hipGraph_t* pGraph) nogil



    hipError_t hipStreamIsCapturing_spt(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus) nogil



    hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus,unsigned long long * pId) nogil



    hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,hipStreamCaptureStatus * captureStatus_out,unsigned long long * id_out,hipGraph_t* graph_out,hipGraphNode_t ** dependencies_out,int * numDependencies_out) nogil



    hipError_t hipLaunchHostFunc_spt(hipStream_t stream,hipHostFn_t fn,void * userData) nogil
