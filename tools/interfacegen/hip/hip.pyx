# AMD_COPYRIGHT
from libc cimport stdlib
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


cdef class hipDeviceArch_t:
    cdef chip.hipDeviceArch_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipDeviceArch_t from_ptr(chip.hipDeviceArch_t *_ptr, bint owner=False):
        """Factory function to create ``hipDeviceArch_t`` objects from
        given ``chip.hipDeviceArch_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipDeviceArch_t wrapper = hipDeviceArch_t.__new__(hipDeviceArch_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class hipUUID_t:
    cdef chip.hipUUID_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipUUID_t from_ptr(chip.hipUUID_t *_ptr, bint owner=False):
        """Factory function to create ``hipUUID_t`` objects from
        given ``chip.hipUUID_t`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipUUID_t wrapper = hipUUID_t.__new__(hipUUID_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipUUID_t new():
        """Factory function to create hipUUID_t objects with
        newly allocated chip.hipUUID_t"""
        cdef chip.hipUUID_t *_ptr = <chip.hipUUID_t *>stdlib.malloc(sizeof(chip.hipUUID_t))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipUUID_t.from_ptr(_ptr, owner=True)



cdef class hipDeviceProp_t:
    cdef chip.hipDeviceProp_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipDeviceProp_t from_ptr(chip.hipDeviceProp_t *_ptr, bint owner=False):
        """Factory function to create ``hipDeviceProp_t`` objects from
        given ``chip.hipDeviceProp_t`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipDeviceProp_t wrapper = hipDeviceProp_t.__new__(hipDeviceProp_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipDeviceProp_t new():
        """Factory function to create hipDeviceProp_t objects with
        newly allocated chip.hipDeviceProp_t"""
        cdef chip.hipDeviceProp_t *_ptr = <chip.hipDeviceProp_t *>stdlib.malloc(sizeof(chip.hipDeviceProp_t))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipDeviceProp_t.from_ptr(_ptr, owner=True)


class hipMemoryType(enum.IntEnum):
    hipMemoryTypeHost = chip.hipMemoryTypeHost
    hipMemoryTypeDevice = chip.hipMemoryTypeDevice
    hipMemoryTypeArray = chip.hipMemoryTypeArray
    hipMemoryTypeUnified = chip.hipMemoryTypeUnified
    hipMemoryTypeManaged = chip.hipMemoryTypeManaged


cdef class hipPointerAttribute_t:
    cdef chip.hipPointerAttribute_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipPointerAttribute_t from_ptr(chip.hipPointerAttribute_t *_ptr, bint owner=False):
        """Factory function to create ``hipPointerAttribute_t`` objects from
        given ``chip.hipPointerAttribute_t`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipPointerAttribute_t wrapper = hipPointerAttribute_t.__new__(hipPointerAttribute_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipPointerAttribute_t new():
        """Factory function to create hipPointerAttribute_t objects with
        newly allocated chip.hipPointerAttribute_t"""
        cdef chip.hipPointerAttribute_t *_ptr = <chip.hipPointerAttribute_t *>stdlib.malloc(sizeof(chip.hipPointerAttribute_t))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipPointerAttribute_t.from_ptr(_ptr, owner=True)


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


cdef class hipChannelFormatDesc:
    cdef chip.hipChannelFormatDesc* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipChannelFormatDesc from_ptr(chip.hipChannelFormatDesc *_ptr, bint owner=False):
        """Factory function to create ``hipChannelFormatDesc`` objects from
        given ``chip.hipChannelFormatDesc`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipChannelFormatDesc wrapper = hipChannelFormatDesc.__new__(hipChannelFormatDesc)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipChannelFormatDesc new():
        """Factory function to create hipChannelFormatDesc objects with
        newly allocated chip.hipChannelFormatDesc"""
        cdef chip.hipChannelFormatDesc *_ptr = <chip.hipChannelFormatDesc *>stdlib.malloc(sizeof(chip.hipChannelFormatDesc))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipChannelFormatDesc.from_ptr(_ptr, owner=True)


class hipArray_Format(enum.IntEnum):
    HIP_AD_FORMAT_UNSIGNED_INT8 = chip.HIP_AD_FORMAT_UNSIGNED_INT8
    HIP_AD_FORMAT_UNSIGNED_INT16 = chip.HIP_AD_FORMAT_UNSIGNED_INT16
    HIP_AD_FORMAT_UNSIGNED_INT32 = chip.HIP_AD_FORMAT_UNSIGNED_INT32
    HIP_AD_FORMAT_SIGNED_INT8 = chip.HIP_AD_FORMAT_SIGNED_INT8
    HIP_AD_FORMAT_SIGNED_INT16 = chip.HIP_AD_FORMAT_SIGNED_INT16
    HIP_AD_FORMAT_SIGNED_INT32 = chip.HIP_AD_FORMAT_SIGNED_INT32
    HIP_AD_FORMAT_HALF = chip.HIP_AD_FORMAT_HALF
    HIP_AD_FORMAT_FLOAT = chip.HIP_AD_FORMAT_FLOAT


cdef class HIP_ARRAY_DESCRIPTOR:
    cdef chip.HIP_ARRAY_DESCRIPTOR* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_ARRAY_DESCRIPTOR from_ptr(chip.HIP_ARRAY_DESCRIPTOR *_ptr, bint owner=False):
        """Factory function to create ``HIP_ARRAY_DESCRIPTOR`` objects from
        given ``chip.HIP_ARRAY_DESCRIPTOR`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_ARRAY_DESCRIPTOR wrapper = HIP_ARRAY_DESCRIPTOR.__new__(HIP_ARRAY_DESCRIPTOR)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_ARRAY_DESCRIPTOR new():
        """Factory function to create HIP_ARRAY_DESCRIPTOR objects with
        newly allocated chip.HIP_ARRAY_DESCRIPTOR"""
        cdef chip.HIP_ARRAY_DESCRIPTOR *_ptr = <chip.HIP_ARRAY_DESCRIPTOR *>stdlib.malloc(sizeof(chip.HIP_ARRAY_DESCRIPTOR))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_ARRAY_DESCRIPTOR.from_ptr(_ptr, owner=True)



cdef class HIP_ARRAY3D_DESCRIPTOR:
    cdef chip.HIP_ARRAY3D_DESCRIPTOR* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_ARRAY3D_DESCRIPTOR from_ptr(chip.HIP_ARRAY3D_DESCRIPTOR *_ptr, bint owner=False):
        """Factory function to create ``HIP_ARRAY3D_DESCRIPTOR`` objects from
        given ``chip.HIP_ARRAY3D_DESCRIPTOR`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_ARRAY3D_DESCRIPTOR wrapper = HIP_ARRAY3D_DESCRIPTOR.__new__(HIP_ARRAY3D_DESCRIPTOR)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_ARRAY3D_DESCRIPTOR new():
        """Factory function to create HIP_ARRAY3D_DESCRIPTOR objects with
        newly allocated chip.HIP_ARRAY3D_DESCRIPTOR"""
        cdef chip.HIP_ARRAY3D_DESCRIPTOR *_ptr = <chip.HIP_ARRAY3D_DESCRIPTOR *>stdlib.malloc(sizeof(chip.HIP_ARRAY3D_DESCRIPTOR))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_ARRAY3D_DESCRIPTOR.from_ptr(_ptr, owner=True)



cdef class hipArray:
    cdef chip.hipArray* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipArray from_ptr(chip.hipArray *_ptr, bint owner=False):
        """Factory function to create ``hipArray`` objects from
        given ``chip.hipArray`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipArray wrapper = hipArray.__new__(hipArray)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipArray new():
        """Factory function to create hipArray objects with
        newly allocated chip.hipArray"""
        cdef chip.hipArray *_ptr = <chip.hipArray *>stdlib.malloc(sizeof(chip.hipArray))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipArray.from_ptr(_ptr, owner=True)



cdef class hip_Memcpy2D:
    cdef chip.hip_Memcpy2D* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hip_Memcpy2D from_ptr(chip.hip_Memcpy2D *_ptr, bint owner=False):
        """Factory function to create ``hip_Memcpy2D`` objects from
        given ``chip.hip_Memcpy2D`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hip_Memcpy2D wrapper = hip_Memcpy2D.__new__(hip_Memcpy2D)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hip_Memcpy2D new():
        """Factory function to create hip_Memcpy2D objects with
        newly allocated chip.hip_Memcpy2D"""
        cdef chip.hip_Memcpy2D *_ptr = <chip.hip_Memcpy2D *>stdlib.malloc(sizeof(chip.hip_Memcpy2D))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hip_Memcpy2D.from_ptr(_ptr, owner=True)


hipArray_t = hipArray

hiparray = hipArray_t

hipArray_const_t = hipArray


cdef class hipMipmappedArray:
    cdef chip.hipMipmappedArray* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipMipmappedArray from_ptr(chip.hipMipmappedArray *_ptr, bint owner=False):
        """Factory function to create ``hipMipmappedArray`` objects from
        given ``chip.hipMipmappedArray`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipMipmappedArray wrapper = hipMipmappedArray.__new__(hipMipmappedArray)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipMipmappedArray new():
        """Factory function to create hipMipmappedArray objects with
        newly allocated chip.hipMipmappedArray"""
        cdef chip.hipMipmappedArray *_ptr = <chip.hipMipmappedArray *>stdlib.malloc(sizeof(chip.hipMipmappedArray))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipMipmappedArray.from_ptr(_ptr, owner=True)


hipMipmappedArray_t = hipMipmappedArray

hipMipmappedArray_const_t = hipMipmappedArray

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


cdef class HIP_TEXTURE_DESC_st:
    cdef chip.HIP_TEXTURE_DESC_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_TEXTURE_DESC_st from_ptr(chip.HIP_TEXTURE_DESC_st *_ptr, bint owner=False):
        """Factory function to create ``HIP_TEXTURE_DESC_st`` objects from
        given ``chip.HIP_TEXTURE_DESC_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_TEXTURE_DESC_st wrapper = HIP_TEXTURE_DESC_st.__new__(HIP_TEXTURE_DESC_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_TEXTURE_DESC_st new():
        """Factory function to create HIP_TEXTURE_DESC_st objects with
        newly allocated chip.HIP_TEXTURE_DESC_st"""
        cdef chip.HIP_TEXTURE_DESC_st *_ptr = <chip.HIP_TEXTURE_DESC_st *>stdlib.malloc(sizeof(chip.HIP_TEXTURE_DESC_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_TEXTURE_DESC_st.from_ptr(_ptr, owner=True)


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


cdef class hipResourceDesc_union_0_struct_0:
    cdef chip.hipResourceDesc_union_0_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipResourceDesc_union_0_struct_0 from_ptr(chip.hipResourceDesc_union_0_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``hipResourceDesc_union_0_struct_0`` objects from
        given ``chip.hipResourceDesc_union_0_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipResourceDesc_union_0_struct_0 wrapper = hipResourceDesc_union_0_struct_0.__new__(hipResourceDesc_union_0_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipResourceDesc_union_0_struct_0 new():
        """Factory function to create hipResourceDesc_union_0_struct_0 objects with
        newly allocated chip.hipResourceDesc_union_0_struct_0"""
        cdef chip.hipResourceDesc_union_0_struct_0 *_ptr = <chip.hipResourceDesc_union_0_struct_0 *>stdlib.malloc(sizeof(chip.hipResourceDesc_union_0_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipResourceDesc_union_0_struct_0.from_ptr(_ptr, owner=True)



cdef class hipResourceDesc_union_0_struct_1:
    cdef chip.hipResourceDesc_union_0_struct_1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipResourceDesc_union_0_struct_1 from_ptr(chip.hipResourceDesc_union_0_struct_1 *_ptr, bint owner=False):
        """Factory function to create ``hipResourceDesc_union_0_struct_1`` objects from
        given ``chip.hipResourceDesc_union_0_struct_1`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipResourceDesc_union_0_struct_1 wrapper = hipResourceDesc_union_0_struct_1.__new__(hipResourceDesc_union_0_struct_1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipResourceDesc_union_0_struct_1 new():
        """Factory function to create hipResourceDesc_union_0_struct_1 objects with
        newly allocated chip.hipResourceDesc_union_0_struct_1"""
        cdef chip.hipResourceDesc_union_0_struct_1 *_ptr = <chip.hipResourceDesc_union_0_struct_1 *>stdlib.malloc(sizeof(chip.hipResourceDesc_union_0_struct_1))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipResourceDesc_union_0_struct_1.from_ptr(_ptr, owner=True)



cdef class hipResourceDesc_union_0_struct_2:
    cdef chip.hipResourceDesc_union_0_struct_2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipResourceDesc_union_0_struct_2 from_ptr(chip.hipResourceDesc_union_0_struct_2 *_ptr, bint owner=False):
        """Factory function to create ``hipResourceDesc_union_0_struct_2`` objects from
        given ``chip.hipResourceDesc_union_0_struct_2`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipResourceDesc_union_0_struct_2 wrapper = hipResourceDesc_union_0_struct_2.__new__(hipResourceDesc_union_0_struct_2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipResourceDesc_union_0_struct_2 new():
        """Factory function to create hipResourceDesc_union_0_struct_2 objects with
        newly allocated chip.hipResourceDesc_union_0_struct_2"""
        cdef chip.hipResourceDesc_union_0_struct_2 *_ptr = <chip.hipResourceDesc_union_0_struct_2 *>stdlib.malloc(sizeof(chip.hipResourceDesc_union_0_struct_2))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipResourceDesc_union_0_struct_2.from_ptr(_ptr, owner=True)



cdef class hipResourceDesc_union_0_struct_3:
    cdef chip.hipResourceDesc_union_0_struct_3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipResourceDesc_union_0_struct_3 from_ptr(chip.hipResourceDesc_union_0_struct_3 *_ptr, bint owner=False):
        """Factory function to create ``hipResourceDesc_union_0_struct_3`` objects from
        given ``chip.hipResourceDesc_union_0_struct_3`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipResourceDesc_union_0_struct_3 wrapper = hipResourceDesc_union_0_struct_3.__new__(hipResourceDesc_union_0_struct_3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipResourceDesc_union_0_struct_3 new():
        """Factory function to create hipResourceDesc_union_0_struct_3 objects with
        newly allocated chip.hipResourceDesc_union_0_struct_3"""
        cdef chip.hipResourceDesc_union_0_struct_3 *_ptr = <chip.hipResourceDesc_union_0_struct_3 *>stdlib.malloc(sizeof(chip.hipResourceDesc_union_0_struct_3))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipResourceDesc_union_0_struct_3.from_ptr(_ptr, owner=True)



cdef class hipResourceDesc_union_0:
    cdef chip.hipResourceDesc_union_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipResourceDesc_union_0 from_ptr(chip.hipResourceDesc_union_0 *_ptr, bint owner=False):
        """Factory function to create ``hipResourceDesc_union_0`` objects from
        given ``chip.hipResourceDesc_union_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipResourceDesc_union_0 wrapper = hipResourceDesc_union_0.__new__(hipResourceDesc_union_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipResourceDesc_union_0 new():
        """Factory function to create hipResourceDesc_union_0 objects with
        newly allocated chip.hipResourceDesc_union_0"""
        cdef chip.hipResourceDesc_union_0 *_ptr = <chip.hipResourceDesc_union_0 *>stdlib.malloc(sizeof(chip.hipResourceDesc_union_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipResourceDesc_union_0.from_ptr(_ptr, owner=True)



cdef class hipResourceDesc:
    cdef chip.hipResourceDesc* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipResourceDesc from_ptr(chip.hipResourceDesc *_ptr, bint owner=False):
        """Factory function to create ``hipResourceDesc`` objects from
        given ``chip.hipResourceDesc`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipResourceDesc wrapper = hipResourceDesc.__new__(hipResourceDesc)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipResourceDesc new():
        """Factory function to create hipResourceDesc objects with
        newly allocated chip.hipResourceDesc"""
        cdef chip.hipResourceDesc *_ptr = <chip.hipResourceDesc *>stdlib.malloc(sizeof(chip.hipResourceDesc))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipResourceDesc.from_ptr(_ptr, owner=True)



cdef class HIP_RESOURCE_DESC_st_union_0_struct_0:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_0 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_DESC_st_union_0_struct_0`` objects from
        given ``chip.HIP_RESOURCE_DESC_st_union_0_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_DESC_st_union_0_struct_0 wrapper = HIP_RESOURCE_DESC_st_union_0_struct_0.__new__(HIP_RESOURCE_DESC_st_union_0_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_0 new():
        """Factory function to create HIP_RESOURCE_DESC_st_union_0_struct_0 objects with
        newly allocated chip.HIP_RESOURCE_DESC_st_union_0_struct_0"""
        cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_0 *_ptr = <chip.HIP_RESOURCE_DESC_st_union_0_struct_0 *>stdlib.malloc(sizeof(chip.HIP_RESOURCE_DESC_st_union_0_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_RESOURCE_DESC_st_union_0_struct_0.from_ptr(_ptr, owner=True)



cdef class HIP_RESOURCE_DESC_st_union_0_struct_1:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_1 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_1 *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_DESC_st_union_0_struct_1`` objects from
        given ``chip.HIP_RESOURCE_DESC_st_union_0_struct_1`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_DESC_st_union_0_struct_1 wrapper = HIP_RESOURCE_DESC_st_union_0_struct_1.__new__(HIP_RESOURCE_DESC_st_union_0_struct_1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_1 new():
        """Factory function to create HIP_RESOURCE_DESC_st_union_0_struct_1 objects with
        newly allocated chip.HIP_RESOURCE_DESC_st_union_0_struct_1"""
        cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_1 *_ptr = <chip.HIP_RESOURCE_DESC_st_union_0_struct_1 *>stdlib.malloc(sizeof(chip.HIP_RESOURCE_DESC_st_union_0_struct_1))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_RESOURCE_DESC_st_union_0_struct_1.from_ptr(_ptr, owner=True)



cdef class HIP_RESOURCE_DESC_st_union_0_struct_2:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_2 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_2 *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_DESC_st_union_0_struct_2`` objects from
        given ``chip.HIP_RESOURCE_DESC_st_union_0_struct_2`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_DESC_st_union_0_struct_2 wrapper = HIP_RESOURCE_DESC_st_union_0_struct_2.__new__(HIP_RESOURCE_DESC_st_union_0_struct_2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_2 new():
        """Factory function to create HIP_RESOURCE_DESC_st_union_0_struct_2 objects with
        newly allocated chip.HIP_RESOURCE_DESC_st_union_0_struct_2"""
        cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_2 *_ptr = <chip.HIP_RESOURCE_DESC_st_union_0_struct_2 *>stdlib.malloc(sizeof(chip.HIP_RESOURCE_DESC_st_union_0_struct_2))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_RESOURCE_DESC_st_union_0_struct_2.from_ptr(_ptr, owner=True)



cdef class HIP_RESOURCE_DESC_st_union_0_struct_3:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_3 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_3 *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_DESC_st_union_0_struct_3`` objects from
        given ``chip.HIP_RESOURCE_DESC_st_union_0_struct_3`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_DESC_st_union_0_struct_3 wrapper = HIP_RESOURCE_DESC_st_union_0_struct_3.__new__(HIP_RESOURCE_DESC_st_union_0_struct_3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_3 new():
        """Factory function to create HIP_RESOURCE_DESC_st_union_0_struct_3 objects with
        newly allocated chip.HIP_RESOURCE_DESC_st_union_0_struct_3"""
        cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_3 *_ptr = <chip.HIP_RESOURCE_DESC_st_union_0_struct_3 *>stdlib.malloc(sizeof(chip.HIP_RESOURCE_DESC_st_union_0_struct_3))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_RESOURCE_DESC_st_union_0_struct_3.from_ptr(_ptr, owner=True)



cdef class HIP_RESOURCE_DESC_st_union_0_struct_4:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_4 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_4 *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_DESC_st_union_0_struct_4`` objects from
        given ``chip.HIP_RESOURCE_DESC_st_union_0_struct_4`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_DESC_st_union_0_struct_4 wrapper = HIP_RESOURCE_DESC_st_union_0_struct_4.__new__(HIP_RESOURCE_DESC_st_union_0_struct_4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_4 new():
        """Factory function to create HIP_RESOURCE_DESC_st_union_0_struct_4 objects with
        newly allocated chip.HIP_RESOURCE_DESC_st_union_0_struct_4"""
        cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_4 *_ptr = <chip.HIP_RESOURCE_DESC_st_union_0_struct_4 *>stdlib.malloc(sizeof(chip.HIP_RESOURCE_DESC_st_union_0_struct_4))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_RESOURCE_DESC_st_union_0_struct_4.from_ptr(_ptr, owner=True)



cdef class HIP_RESOURCE_DESC_st_union_0:
    cdef chip.HIP_RESOURCE_DESC_st_union_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0 *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_DESC_st_union_0`` objects from
        given ``chip.HIP_RESOURCE_DESC_st_union_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_DESC_st_union_0 wrapper = HIP_RESOURCE_DESC_st_union_0.__new__(HIP_RESOURCE_DESC_st_union_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0 new():
        """Factory function to create HIP_RESOURCE_DESC_st_union_0 objects with
        newly allocated chip.HIP_RESOURCE_DESC_st_union_0"""
        cdef chip.HIP_RESOURCE_DESC_st_union_0 *_ptr = <chip.HIP_RESOURCE_DESC_st_union_0 *>stdlib.malloc(sizeof(chip.HIP_RESOURCE_DESC_st_union_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_RESOURCE_DESC_st_union_0.from_ptr(_ptr, owner=True)



cdef class HIP_RESOURCE_DESC_st:
    cdef chip.HIP_RESOURCE_DESC_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_DESC_st from_ptr(chip.HIP_RESOURCE_DESC_st *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_DESC_st`` objects from
        given ``chip.HIP_RESOURCE_DESC_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_DESC_st wrapper = HIP_RESOURCE_DESC_st.__new__(HIP_RESOURCE_DESC_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_RESOURCE_DESC_st new():
        """Factory function to create HIP_RESOURCE_DESC_st objects with
        newly allocated chip.HIP_RESOURCE_DESC_st"""
        cdef chip.HIP_RESOURCE_DESC_st *_ptr = <chip.HIP_RESOURCE_DESC_st *>stdlib.malloc(sizeof(chip.HIP_RESOURCE_DESC_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_RESOURCE_DESC_st.from_ptr(_ptr, owner=True)



cdef class hipResourceViewDesc:
    cdef chip.hipResourceViewDesc* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipResourceViewDesc from_ptr(chip.hipResourceViewDesc *_ptr, bint owner=False):
        """Factory function to create ``hipResourceViewDesc`` objects from
        given ``chip.hipResourceViewDesc`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipResourceViewDesc wrapper = hipResourceViewDesc.__new__(hipResourceViewDesc)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipResourceViewDesc new():
        """Factory function to create hipResourceViewDesc objects with
        newly allocated chip.hipResourceViewDesc"""
        cdef chip.hipResourceViewDesc *_ptr = <chip.hipResourceViewDesc *>stdlib.malloc(sizeof(chip.hipResourceViewDesc))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipResourceViewDesc.from_ptr(_ptr, owner=True)



cdef class HIP_RESOURCE_VIEW_DESC_st:
    cdef chip.HIP_RESOURCE_VIEW_DESC_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_VIEW_DESC_st from_ptr(chip.HIP_RESOURCE_VIEW_DESC_st *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_VIEW_DESC_st`` objects from
        given ``chip.HIP_RESOURCE_VIEW_DESC_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_VIEW_DESC_st wrapper = HIP_RESOURCE_VIEW_DESC_st.__new__(HIP_RESOURCE_VIEW_DESC_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_RESOURCE_VIEW_DESC_st new():
        """Factory function to create HIP_RESOURCE_VIEW_DESC_st objects with
        newly allocated chip.HIP_RESOURCE_VIEW_DESC_st"""
        cdef chip.HIP_RESOURCE_VIEW_DESC_st *_ptr = <chip.HIP_RESOURCE_VIEW_DESC_st *>stdlib.malloc(sizeof(chip.HIP_RESOURCE_VIEW_DESC_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_RESOURCE_VIEW_DESC_st.from_ptr(_ptr, owner=True)


class hipMemcpyKind(enum.IntEnum):
    hipMemcpyHostToHost = chip.hipMemcpyHostToHost
    hipMemcpyHostToDevice = chip.hipMemcpyHostToDevice
    hipMemcpyDeviceToHost = chip.hipMemcpyDeviceToHost
    hipMemcpyDeviceToDevice = chip.hipMemcpyDeviceToDevice
    hipMemcpyDefault = chip.hipMemcpyDefault


cdef class hipPitchedPtr:
    cdef chip.hipPitchedPtr* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipPitchedPtr from_ptr(chip.hipPitchedPtr *_ptr, bint owner=False):
        """Factory function to create ``hipPitchedPtr`` objects from
        given ``chip.hipPitchedPtr`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipPitchedPtr wrapper = hipPitchedPtr.__new__(hipPitchedPtr)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipPitchedPtr new():
        """Factory function to create hipPitchedPtr objects with
        newly allocated chip.hipPitchedPtr"""
        cdef chip.hipPitchedPtr *_ptr = <chip.hipPitchedPtr *>stdlib.malloc(sizeof(chip.hipPitchedPtr))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipPitchedPtr.from_ptr(_ptr, owner=True)



cdef class hipExtent:
    cdef chip.hipExtent* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExtent from_ptr(chip.hipExtent *_ptr, bint owner=False):
        """Factory function to create ``hipExtent`` objects from
        given ``chip.hipExtent`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExtent wrapper = hipExtent.__new__(hipExtent)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExtent new():
        """Factory function to create hipExtent objects with
        newly allocated chip.hipExtent"""
        cdef chip.hipExtent *_ptr = <chip.hipExtent *>stdlib.malloc(sizeof(chip.hipExtent))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExtent.from_ptr(_ptr, owner=True)



cdef class hipPos:
    cdef chip.hipPos* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipPos from_ptr(chip.hipPos *_ptr, bint owner=False):
        """Factory function to create ``hipPos`` objects from
        given ``chip.hipPos`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipPos wrapper = hipPos.__new__(hipPos)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipPos new():
        """Factory function to create hipPos objects with
        newly allocated chip.hipPos"""
        cdef chip.hipPos *_ptr = <chip.hipPos *>stdlib.malloc(sizeof(chip.hipPos))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipPos.from_ptr(_ptr, owner=True)



cdef class hipMemcpy3DParms:
    cdef chip.hipMemcpy3DParms* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipMemcpy3DParms from_ptr(chip.hipMemcpy3DParms *_ptr, bint owner=False):
        """Factory function to create ``hipMemcpy3DParms`` objects from
        given ``chip.hipMemcpy3DParms`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipMemcpy3DParms wrapper = hipMemcpy3DParms.__new__(hipMemcpy3DParms)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipMemcpy3DParms new():
        """Factory function to create hipMemcpy3DParms objects with
        newly allocated chip.hipMemcpy3DParms"""
        cdef chip.hipMemcpy3DParms *_ptr = <chip.hipMemcpy3DParms *>stdlib.malloc(sizeof(chip.hipMemcpy3DParms))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipMemcpy3DParms.from_ptr(_ptr, owner=True)



cdef class HIP_MEMCPY3D:
    cdef chip.HIP_MEMCPY3D* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_MEMCPY3D from_ptr(chip.HIP_MEMCPY3D *_ptr, bint owner=False):
        """Factory function to create ``HIP_MEMCPY3D`` objects from
        given ``chip.HIP_MEMCPY3D`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_MEMCPY3D wrapper = HIP_MEMCPY3D.__new__(HIP_MEMCPY3D)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef HIP_MEMCPY3D new():
        """Factory function to create HIP_MEMCPY3D objects with
        newly allocated chip.HIP_MEMCPY3D"""
        cdef chip.HIP_MEMCPY3D *_ptr = <chip.HIP_MEMCPY3D *>stdlib.malloc(sizeof(chip.HIP_MEMCPY3D))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return HIP_MEMCPY3D.from_ptr(_ptr, owner=True)


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


cdef class uchar1:
    cdef chip.uchar1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef uchar1 from_ptr(chip.uchar1 *_ptr, bint owner=False):
        """Factory function to create ``uchar1`` objects from
        given ``chip.uchar1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef uchar1 wrapper = uchar1.__new__(uchar1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class uchar2:
    cdef chip.uchar2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef uchar2 from_ptr(chip.uchar2 *_ptr, bint owner=False):
        """Factory function to create ``uchar2`` objects from
        given ``chip.uchar2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef uchar2 wrapper = uchar2.__new__(uchar2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class uchar3:
    cdef chip.uchar3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef uchar3 from_ptr(chip.uchar3 *_ptr, bint owner=False):
        """Factory function to create ``uchar3`` objects from
        given ``chip.uchar3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef uchar3 wrapper = uchar3.__new__(uchar3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class uchar4:
    cdef chip.uchar4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef uchar4 from_ptr(chip.uchar4 *_ptr, bint owner=False):
        """Factory function to create ``uchar4`` objects from
        given ``chip.uchar4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef uchar4 wrapper = uchar4.__new__(uchar4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class char1:
    cdef chip.char1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef char1 from_ptr(chip.char1 *_ptr, bint owner=False):
        """Factory function to create ``char1`` objects from
        given ``chip.char1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef char1 wrapper = char1.__new__(char1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class char2:
    cdef chip.char2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef char2 from_ptr(chip.char2 *_ptr, bint owner=False):
        """Factory function to create ``char2`` objects from
        given ``chip.char2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef char2 wrapper = char2.__new__(char2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class char3:
    cdef chip.char3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef char3 from_ptr(chip.char3 *_ptr, bint owner=False):
        """Factory function to create ``char3`` objects from
        given ``chip.char3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef char3 wrapper = char3.__new__(char3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class char4:
    cdef chip.char4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef char4 from_ptr(chip.char4 *_ptr, bint owner=False):
        """Factory function to create ``char4`` objects from
        given ``chip.char4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef char4 wrapper = char4.__new__(char4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ushort1:
    cdef chip.ushort1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ushort1 from_ptr(chip.ushort1 *_ptr, bint owner=False):
        """Factory function to create ``ushort1`` objects from
        given ``chip.ushort1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ushort1 wrapper = ushort1.__new__(ushort1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ushort2:
    cdef chip.ushort2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ushort2 from_ptr(chip.ushort2 *_ptr, bint owner=False):
        """Factory function to create ``ushort2`` objects from
        given ``chip.ushort2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ushort2 wrapper = ushort2.__new__(ushort2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ushort3:
    cdef chip.ushort3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ushort3 from_ptr(chip.ushort3 *_ptr, bint owner=False):
        """Factory function to create ``ushort3`` objects from
        given ``chip.ushort3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ushort3 wrapper = ushort3.__new__(ushort3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ushort4:
    cdef chip.ushort4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ushort4 from_ptr(chip.ushort4 *_ptr, bint owner=False):
        """Factory function to create ``ushort4`` objects from
        given ``chip.ushort4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ushort4 wrapper = ushort4.__new__(ushort4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class short1:
    cdef chip.short1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef short1 from_ptr(chip.short1 *_ptr, bint owner=False):
        """Factory function to create ``short1`` objects from
        given ``chip.short1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef short1 wrapper = short1.__new__(short1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class short2:
    cdef chip.short2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef short2 from_ptr(chip.short2 *_ptr, bint owner=False):
        """Factory function to create ``short2`` objects from
        given ``chip.short2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef short2 wrapper = short2.__new__(short2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class short3:
    cdef chip.short3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef short3 from_ptr(chip.short3 *_ptr, bint owner=False):
        """Factory function to create ``short3`` objects from
        given ``chip.short3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef short3 wrapper = short3.__new__(short3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class short4:
    cdef chip.short4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef short4 from_ptr(chip.short4 *_ptr, bint owner=False):
        """Factory function to create ``short4`` objects from
        given ``chip.short4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef short4 wrapper = short4.__new__(short4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class uint1:
    cdef chip.uint1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef uint1 from_ptr(chip.uint1 *_ptr, bint owner=False):
        """Factory function to create ``uint1`` objects from
        given ``chip.uint1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef uint1 wrapper = uint1.__new__(uint1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class uint2:
    cdef chip.uint2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef uint2 from_ptr(chip.uint2 *_ptr, bint owner=False):
        """Factory function to create ``uint2`` objects from
        given ``chip.uint2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef uint2 wrapper = uint2.__new__(uint2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class uint3:
    cdef chip.uint3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef uint3 from_ptr(chip.uint3 *_ptr, bint owner=False):
        """Factory function to create ``uint3`` objects from
        given ``chip.uint3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef uint3 wrapper = uint3.__new__(uint3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class uint4:
    cdef chip.uint4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef uint4 from_ptr(chip.uint4 *_ptr, bint owner=False):
        """Factory function to create ``uint4`` objects from
        given ``chip.uint4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef uint4 wrapper = uint4.__new__(uint4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class int1:
    cdef chip.int1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef int1 from_ptr(chip.int1 *_ptr, bint owner=False):
        """Factory function to create ``int1`` objects from
        given ``chip.int1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef int1 wrapper = int1.__new__(int1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class int2:
    cdef chip.int2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef int2 from_ptr(chip.int2 *_ptr, bint owner=False):
        """Factory function to create ``int2`` objects from
        given ``chip.int2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef int2 wrapper = int2.__new__(int2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class int3:
    cdef chip.int3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef int3 from_ptr(chip.int3 *_ptr, bint owner=False):
        """Factory function to create ``int3`` objects from
        given ``chip.int3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef int3 wrapper = int3.__new__(int3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class int4:
    cdef chip.int4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef int4 from_ptr(chip.int4 *_ptr, bint owner=False):
        """Factory function to create ``int4`` objects from
        given ``chip.int4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef int4 wrapper = int4.__new__(int4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ulong1:
    cdef chip.ulong1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ulong1 from_ptr(chip.ulong1 *_ptr, bint owner=False):
        """Factory function to create ``ulong1`` objects from
        given ``chip.ulong1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ulong1 wrapper = ulong1.__new__(ulong1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ulong2:
    cdef chip.ulong2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ulong2 from_ptr(chip.ulong2 *_ptr, bint owner=False):
        """Factory function to create ``ulong2`` objects from
        given ``chip.ulong2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ulong2 wrapper = ulong2.__new__(ulong2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ulong3:
    cdef chip.ulong3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ulong3 from_ptr(chip.ulong3 *_ptr, bint owner=False):
        """Factory function to create ``ulong3`` objects from
        given ``chip.ulong3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ulong3 wrapper = ulong3.__new__(ulong3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ulong4:
    cdef chip.ulong4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ulong4 from_ptr(chip.ulong4 *_ptr, bint owner=False):
        """Factory function to create ``ulong4`` objects from
        given ``chip.ulong4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ulong4 wrapper = ulong4.__new__(ulong4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class long1:
    cdef chip.long1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef long1 from_ptr(chip.long1 *_ptr, bint owner=False):
        """Factory function to create ``long1`` objects from
        given ``chip.long1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef long1 wrapper = long1.__new__(long1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class long2:
    cdef chip.long2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef long2 from_ptr(chip.long2 *_ptr, bint owner=False):
        """Factory function to create ``long2`` objects from
        given ``chip.long2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef long2 wrapper = long2.__new__(long2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class long3:
    cdef chip.long3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef long3 from_ptr(chip.long3 *_ptr, bint owner=False):
        """Factory function to create ``long3`` objects from
        given ``chip.long3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef long3 wrapper = long3.__new__(long3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class long4:
    cdef chip.long4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef long4 from_ptr(chip.long4 *_ptr, bint owner=False):
        """Factory function to create ``long4`` objects from
        given ``chip.long4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef long4 wrapper = long4.__new__(long4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ulonglong1:
    cdef chip.ulonglong1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ulonglong1 from_ptr(chip.ulonglong1 *_ptr, bint owner=False):
        """Factory function to create ``ulonglong1`` objects from
        given ``chip.ulonglong1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ulonglong1 wrapper = ulonglong1.__new__(ulonglong1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ulonglong2:
    cdef chip.ulonglong2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ulonglong2 from_ptr(chip.ulonglong2 *_ptr, bint owner=False):
        """Factory function to create ``ulonglong2`` objects from
        given ``chip.ulonglong2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ulonglong2 wrapper = ulonglong2.__new__(ulonglong2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ulonglong3:
    cdef chip.ulonglong3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ulonglong3 from_ptr(chip.ulonglong3 *_ptr, bint owner=False):
        """Factory function to create ``ulonglong3`` objects from
        given ``chip.ulonglong3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ulonglong3 wrapper = ulonglong3.__new__(ulonglong3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class ulonglong4:
    cdef chip.ulonglong4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ulonglong4 from_ptr(chip.ulonglong4 *_ptr, bint owner=False):
        """Factory function to create ``ulonglong4`` objects from
        given ``chip.ulonglong4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ulonglong4 wrapper = ulonglong4.__new__(ulonglong4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class longlong1:
    cdef chip.longlong1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef longlong1 from_ptr(chip.longlong1 *_ptr, bint owner=False):
        """Factory function to create ``longlong1`` objects from
        given ``chip.longlong1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef longlong1 wrapper = longlong1.__new__(longlong1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class longlong2:
    cdef chip.longlong2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef longlong2 from_ptr(chip.longlong2 *_ptr, bint owner=False):
        """Factory function to create ``longlong2`` objects from
        given ``chip.longlong2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef longlong2 wrapper = longlong2.__new__(longlong2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class longlong3:
    cdef chip.longlong3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef longlong3 from_ptr(chip.longlong3 *_ptr, bint owner=False):
        """Factory function to create ``longlong3`` objects from
        given ``chip.longlong3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef longlong3 wrapper = longlong3.__new__(longlong3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class longlong4:
    cdef chip.longlong4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef longlong4 from_ptr(chip.longlong4 *_ptr, bint owner=False):
        """Factory function to create ``longlong4`` objects from
        given ``chip.longlong4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef longlong4 wrapper = longlong4.__new__(longlong4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class float1:
    cdef chip.float1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef float1 from_ptr(chip.float1 *_ptr, bint owner=False):
        """Factory function to create ``float1`` objects from
        given ``chip.float1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef float1 wrapper = float1.__new__(float1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class float2:
    cdef chip.float2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef float2 from_ptr(chip.float2 *_ptr, bint owner=False):
        """Factory function to create ``float2`` objects from
        given ``chip.float2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef float2 wrapper = float2.__new__(float2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class float3:
    cdef chip.float3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef float3 from_ptr(chip.float3 *_ptr, bint owner=False):
        """Factory function to create ``float3`` objects from
        given ``chip.float3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef float3 wrapper = float3.__new__(float3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class float4:
    cdef chip.float4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef float4 from_ptr(chip.float4 *_ptr, bint owner=False):
        """Factory function to create ``float4`` objects from
        given ``chip.float4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef float4 wrapper = float4.__new__(float4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class double1:
    cdef chip.double1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef double1 from_ptr(chip.double1 *_ptr, bint owner=False):
        """Factory function to create ``double1`` objects from
        given ``chip.double1`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef double1 wrapper = double1.__new__(double1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class double2:
    cdef chip.double2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef double2 from_ptr(chip.double2 *_ptr, bint owner=False):
        """Factory function to create ``double2`` objects from
        given ``chip.double2`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef double2 wrapper = double2.__new__(double2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class double3:
    cdef chip.double3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef double3 from_ptr(chip.double3 *_ptr, bint owner=False):
        """Factory function to create ``double3`` objects from
        given ``chip.double3`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef double3 wrapper = double3.__new__(double3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class double4:
    cdef chip.double4* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef double4 from_ptr(chip.double4 *_ptr, bint owner=False):
        """Factory function to create ``double4`` objects from
        given ``chip.double4`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef double4 wrapper = double4.__new__(double4)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class __hip_texture:
    cdef chip.__hip_texture* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef __hip_texture from_ptr(chip.__hip_texture *_ptr, bint owner=False):
        """Factory function to create ``__hip_texture`` objects from
        given ``chip.__hip_texture`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef __hip_texture wrapper = __hip_texture.__new__(__hip_texture)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipTextureObject_t = __hip_texture

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


cdef class textureReference:
    cdef chip.textureReference* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef textureReference from_ptr(chip.textureReference *_ptr, bint owner=False):
        """Factory function to create ``textureReference`` objects from
        given ``chip.textureReference`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef textureReference wrapper = textureReference.__new__(textureReference)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef textureReference new():
        """Factory function to create textureReference objects with
        newly allocated chip.textureReference"""
        cdef chip.textureReference *_ptr = <chip.textureReference *>stdlib.malloc(sizeof(chip.textureReference))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return textureReference.from_ptr(_ptr, owner=True)



cdef class hipTextureDesc:
    cdef chip.hipTextureDesc* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipTextureDesc from_ptr(chip.hipTextureDesc *_ptr, bint owner=False):
        """Factory function to create ``hipTextureDesc`` objects from
        given ``chip.hipTextureDesc`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipTextureDesc wrapper = hipTextureDesc.__new__(hipTextureDesc)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipTextureDesc new():
        """Factory function to create hipTextureDesc objects with
        newly allocated chip.hipTextureDesc"""
        cdef chip.hipTextureDesc *_ptr = <chip.hipTextureDesc *>stdlib.malloc(sizeof(chip.hipTextureDesc))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipTextureDesc.from_ptr(_ptr, owner=True)



cdef class __hip_surface:
    cdef chip.__hip_surface* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef __hip_surface from_ptr(chip.__hip_surface *_ptr, bint owner=False):
        """Factory function to create ``__hip_surface`` objects from
        given ``chip.__hip_surface`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef __hip_surface wrapper = __hip_surface.__new__(__hip_surface)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipSurfaceObject_t = __hip_surface


cdef class surfaceReference:
    cdef chip.surfaceReference* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef surfaceReference from_ptr(chip.surfaceReference *_ptr, bint owner=False):
        """Factory function to create ``surfaceReference`` objects from
        given ``chip.surfaceReference`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef surfaceReference wrapper = surfaceReference.__new__(surfaceReference)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef surfaceReference new():
        """Factory function to create surfaceReference objects with
        newly allocated chip.surfaceReference"""
        cdef chip.surfaceReference *_ptr = <chip.surfaceReference *>stdlib.malloc(sizeof(chip.surfaceReference))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return surfaceReference.from_ptr(_ptr, owner=True)


class hipSurfaceBoundaryMode(enum.IntEnum):
    hipBoundaryModeZero = chip.hipBoundaryModeZero
    hipBoundaryModeTrap = chip.hipBoundaryModeTrap
    hipBoundaryModeClamp = chip.hipBoundaryModeClamp


cdef class ihipCtx_t:
    cdef chip.ihipCtx_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ihipCtx_t from_ptr(chip.ihipCtx_t *_ptr, bint owner=False):
        """Factory function to create ``ihipCtx_t`` objects from
        given ``chip.ihipCtx_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihipCtx_t wrapper = ihipCtx_t.__new__(ihipCtx_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipCtx_t = ihipCtx_t

class hipDeviceP2PAttr(enum.IntEnum):
    hipDevP2PAttrPerformanceRank = chip.hipDevP2PAttrPerformanceRank
    hipDevP2PAttrAccessSupported = chip.hipDevP2PAttrAccessSupported
    hipDevP2PAttrNativeAtomicSupported = chip.hipDevP2PAttrNativeAtomicSupported
    hipDevP2PAttrHipArrayAccessSupported = chip.hipDevP2PAttrHipArrayAccessSupported


cdef class ihipStream_t:
    cdef chip.ihipStream_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ihipStream_t from_ptr(chip.ihipStream_t *_ptr, bint owner=False):
        """Factory function to create ``ihipStream_t`` objects from
        given ``chip.ihipStream_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihipStream_t wrapper = ihipStream_t.__new__(ihipStream_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipStream_t = ihipStream_t


cdef class hipIpcMemHandle_st:
    cdef chip.hipIpcMemHandle_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipIpcMemHandle_st from_ptr(chip.hipIpcMemHandle_st *_ptr, bint owner=False):
        """Factory function to create ``hipIpcMemHandle_st`` objects from
        given ``chip.hipIpcMemHandle_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipIpcMemHandle_st wrapper = hipIpcMemHandle_st.__new__(hipIpcMemHandle_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipIpcMemHandle_st new():
        """Factory function to create hipIpcMemHandle_st objects with
        newly allocated chip.hipIpcMemHandle_st"""
        cdef chip.hipIpcMemHandle_st *_ptr = <chip.hipIpcMemHandle_st *>stdlib.malloc(sizeof(chip.hipIpcMemHandle_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipIpcMemHandle_st.from_ptr(_ptr, owner=True)



cdef class hipIpcEventHandle_st:
    cdef chip.hipIpcEventHandle_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipIpcEventHandle_st from_ptr(chip.hipIpcEventHandle_st *_ptr, bint owner=False):
        """Factory function to create ``hipIpcEventHandle_st`` objects from
        given ``chip.hipIpcEventHandle_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipIpcEventHandle_st wrapper = hipIpcEventHandle_st.__new__(hipIpcEventHandle_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipIpcEventHandle_st new():
        """Factory function to create hipIpcEventHandle_st objects with
        newly allocated chip.hipIpcEventHandle_st"""
        cdef chip.hipIpcEventHandle_st *_ptr = <chip.hipIpcEventHandle_st *>stdlib.malloc(sizeof(chip.hipIpcEventHandle_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipIpcEventHandle_st.from_ptr(_ptr, owner=True)



cdef class ihipModule_t:
    cdef chip.ihipModule_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ihipModule_t from_ptr(chip.ihipModule_t *_ptr, bint owner=False):
        """Factory function to create ``ihipModule_t`` objects from
        given ``chip.ihipModule_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihipModule_t wrapper = ihipModule_t.__new__(ihipModule_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipModule_t = ihipModule_t


cdef class ihipModuleSymbol_t:
    cdef chip.ihipModuleSymbol_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ihipModuleSymbol_t from_ptr(chip.ihipModuleSymbol_t *_ptr, bint owner=False):
        """Factory function to create ``ihipModuleSymbol_t`` objects from
        given ``chip.ihipModuleSymbol_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihipModuleSymbol_t wrapper = ihipModuleSymbol_t.__new__(ihipModuleSymbol_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipFunction_t = ihipModuleSymbol_t


cdef class ihipMemPoolHandle_t:
    cdef chip.ihipMemPoolHandle_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ihipMemPoolHandle_t from_ptr(chip.ihipMemPoolHandle_t *_ptr, bint owner=False):
        """Factory function to create ``ihipMemPoolHandle_t`` objects from
        given ``chip.ihipMemPoolHandle_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihipMemPoolHandle_t wrapper = ihipMemPoolHandle_t.__new__(ihipMemPoolHandle_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipMemPool_t = ihipMemPoolHandle_t


cdef class hipFuncAttributes:
    cdef chip.hipFuncAttributes* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipFuncAttributes from_ptr(chip.hipFuncAttributes *_ptr, bint owner=False):
        """Factory function to create ``hipFuncAttributes`` objects from
        given ``chip.hipFuncAttributes`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipFuncAttributes wrapper = hipFuncAttributes.__new__(hipFuncAttributes)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipFuncAttributes new():
        """Factory function to create hipFuncAttributes objects with
        newly allocated chip.hipFuncAttributes"""
        cdef chip.hipFuncAttributes *_ptr = <chip.hipFuncAttributes *>stdlib.malloc(sizeof(chip.hipFuncAttributes))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipFuncAttributes.from_ptr(_ptr, owner=True)



cdef class ihipEvent_t:
    cdef chip.ihipEvent_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ihipEvent_t from_ptr(chip.ihipEvent_t *_ptr, bint owner=False):
        """Factory function to create ``ihipEvent_t`` objects from
        given ``chip.ihipEvent_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihipEvent_t wrapper = ihipEvent_t.__new__(ihipEvent_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipEvent_t = ihipEvent_t

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


cdef class hipMemLocation:
    cdef chip.hipMemLocation* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipMemLocation from_ptr(chip.hipMemLocation *_ptr, bint owner=False):
        """Factory function to create ``hipMemLocation`` objects from
        given ``chip.hipMemLocation`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipMemLocation wrapper = hipMemLocation.__new__(hipMemLocation)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipMemLocation new():
        """Factory function to create hipMemLocation objects with
        newly allocated chip.hipMemLocation"""
        cdef chip.hipMemLocation *_ptr = <chip.hipMemLocation *>stdlib.malloc(sizeof(chip.hipMemLocation))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipMemLocation.from_ptr(_ptr, owner=True)


class hipMemAccessFlags(enum.IntEnum):
    hipMemAccessFlagsProtNone = chip.hipMemAccessFlagsProtNone
    hipMemAccessFlagsProtRead = chip.hipMemAccessFlagsProtRead
    hipMemAccessFlagsProtReadWrite = chip.hipMemAccessFlagsProtReadWrite


cdef class hipMemAccessDesc:
    cdef chip.hipMemAccessDesc* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipMemAccessDesc from_ptr(chip.hipMemAccessDesc *_ptr, bint owner=False):
        """Factory function to create ``hipMemAccessDesc`` objects from
        given ``chip.hipMemAccessDesc`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipMemAccessDesc wrapper = hipMemAccessDesc.__new__(hipMemAccessDesc)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipMemAccessDesc new():
        """Factory function to create hipMemAccessDesc objects with
        newly allocated chip.hipMemAccessDesc"""
        cdef chip.hipMemAccessDesc *_ptr = <chip.hipMemAccessDesc *>stdlib.malloc(sizeof(chip.hipMemAccessDesc))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipMemAccessDesc.from_ptr(_ptr, owner=True)


class hipMemAllocationType(enum.IntEnum):
    hipMemAllocationTypeInvalid = chip.hipMemAllocationTypeInvalid
    hipMemAllocationTypePinned = chip.hipMemAllocationTypePinned
    hipMemAllocationTypeMax = chip.hipMemAllocationTypeMax

class hipMemAllocationHandleType(enum.IntEnum):
    hipMemHandleTypeNone = chip.hipMemHandleTypeNone
    hipMemHandleTypePosixFileDescriptor = chip.hipMemHandleTypePosixFileDescriptor
    hipMemHandleTypeWin32 = chip.hipMemHandleTypeWin32
    hipMemHandleTypeWin32Kmt = chip.hipMemHandleTypeWin32Kmt


cdef class hipMemPoolProps:
    cdef chip.hipMemPoolProps* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipMemPoolProps from_ptr(chip.hipMemPoolProps *_ptr, bint owner=False):
        """Factory function to create ``hipMemPoolProps`` objects from
        given ``chip.hipMemPoolProps`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipMemPoolProps wrapper = hipMemPoolProps.__new__(hipMemPoolProps)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipMemPoolProps new():
        """Factory function to create hipMemPoolProps objects with
        newly allocated chip.hipMemPoolProps"""
        cdef chip.hipMemPoolProps *_ptr = <chip.hipMemPoolProps *>stdlib.malloc(sizeof(chip.hipMemPoolProps))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipMemPoolProps.from_ptr(_ptr, owner=True)



cdef class hipMemPoolPtrExportData:
    cdef chip.hipMemPoolPtrExportData* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipMemPoolPtrExportData from_ptr(chip.hipMemPoolPtrExportData *_ptr, bint owner=False):
        """Factory function to create ``hipMemPoolPtrExportData`` objects from
        given ``chip.hipMemPoolPtrExportData`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipMemPoolPtrExportData wrapper = hipMemPoolPtrExportData.__new__(hipMemPoolPtrExportData)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipMemPoolPtrExportData new():
        """Factory function to create hipMemPoolPtrExportData objects with
        newly allocated chip.hipMemPoolPtrExportData"""
        cdef chip.hipMemPoolPtrExportData *_ptr = <chip.hipMemPoolPtrExportData *>stdlib.malloc(sizeof(chip.hipMemPoolPtrExportData))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipMemPoolPtrExportData.from_ptr(_ptr, owner=True)


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


cdef class dim3:
    cdef chip.dim3* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef dim3 from_ptr(chip.dim3 *_ptr, bint owner=False):
        """Factory function to create ``dim3`` objects from
        given ``chip.dim3`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef dim3 wrapper = dim3.__new__(dim3)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef dim3 new():
        """Factory function to create dim3 objects with
        newly allocated chip.dim3"""
        cdef chip.dim3 *_ptr = <chip.dim3 *>stdlib.malloc(sizeof(chip.dim3))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return dim3.from_ptr(_ptr, owner=True)



cdef class hipLaunchParams_t:
    cdef chip.hipLaunchParams_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipLaunchParams_t from_ptr(chip.hipLaunchParams_t *_ptr, bint owner=False):
        """Factory function to create ``hipLaunchParams_t`` objects from
        given ``chip.hipLaunchParams_t`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipLaunchParams_t wrapper = hipLaunchParams_t.__new__(hipLaunchParams_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipLaunchParams_t new():
        """Factory function to create hipLaunchParams_t objects with
        newly allocated chip.hipLaunchParams_t"""
        cdef chip.hipLaunchParams_t *_ptr = <chip.hipLaunchParams_t *>stdlib.malloc(sizeof(chip.hipLaunchParams_t))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipLaunchParams_t.from_ptr(_ptr, owner=True)


class hipExternalMemoryHandleType_enum(enum.IntEnum):
    hipExternalMemoryHandleTypeOpaqueFd = chip.hipExternalMemoryHandleTypeOpaqueFd
    hipExternalMemoryHandleTypeOpaqueWin32 = chip.hipExternalMemoryHandleTypeOpaqueWin32
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = chip.hipExternalMemoryHandleTypeOpaqueWin32Kmt
    hipExternalMemoryHandleTypeD3D12Heap = chip.hipExternalMemoryHandleTypeD3D12Heap
    hipExternalMemoryHandleTypeD3D12Resource = chip.hipExternalMemoryHandleTypeD3D12Resource
    hipExternalMemoryHandleTypeD3D11Resource = chip.hipExternalMemoryHandleTypeD3D11Resource
    hipExternalMemoryHandleTypeD3D11ResourceKmt = chip.hipExternalMemoryHandleTypeD3D11ResourceKmt


cdef class hipExternalMemoryHandleDesc_st_union_0_struct_0:
    cdef chip.hipExternalMemoryHandleDesc_st_union_0_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0_struct_0 from_ptr(chip.hipExternalMemoryHandleDesc_st_union_0_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalMemoryHandleDesc_st_union_0_struct_0`` objects from
        given ``chip.hipExternalMemoryHandleDesc_st_union_0_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalMemoryHandleDesc_st_union_0_struct_0 wrapper = hipExternalMemoryHandleDesc_st_union_0_struct_0.__new__(hipExternalMemoryHandleDesc_st_union_0_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0_struct_0 new():
        """Factory function to create hipExternalMemoryHandleDesc_st_union_0_struct_0 objects with
        newly allocated chip.hipExternalMemoryHandleDesc_st_union_0_struct_0"""
        cdef chip.hipExternalMemoryHandleDesc_st_union_0_struct_0 *_ptr = <chip.hipExternalMemoryHandleDesc_st_union_0_struct_0 *>stdlib.malloc(sizeof(chip.hipExternalMemoryHandleDesc_st_union_0_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalMemoryHandleDesc_st_union_0_struct_0.from_ptr(_ptr, owner=True)



cdef class hipExternalMemoryHandleDesc_st_union_0:
    cdef chip.hipExternalMemoryHandleDesc_st_union_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0 from_ptr(chip.hipExternalMemoryHandleDesc_st_union_0 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalMemoryHandleDesc_st_union_0`` objects from
        given ``chip.hipExternalMemoryHandleDesc_st_union_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalMemoryHandleDesc_st_union_0 wrapper = hipExternalMemoryHandleDesc_st_union_0.__new__(hipExternalMemoryHandleDesc_st_union_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0 new():
        """Factory function to create hipExternalMemoryHandleDesc_st_union_0 objects with
        newly allocated chip.hipExternalMemoryHandleDesc_st_union_0"""
        cdef chip.hipExternalMemoryHandleDesc_st_union_0 *_ptr = <chip.hipExternalMemoryHandleDesc_st_union_0 *>stdlib.malloc(sizeof(chip.hipExternalMemoryHandleDesc_st_union_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalMemoryHandleDesc_st_union_0.from_ptr(_ptr, owner=True)



cdef class hipExternalMemoryHandleDesc_st:
    cdef chip.hipExternalMemoryHandleDesc_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalMemoryHandleDesc_st from_ptr(chip.hipExternalMemoryHandleDesc_st *_ptr, bint owner=False):
        """Factory function to create ``hipExternalMemoryHandleDesc_st`` objects from
        given ``chip.hipExternalMemoryHandleDesc_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalMemoryHandleDesc_st wrapper = hipExternalMemoryHandleDesc_st.__new__(hipExternalMemoryHandleDesc_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st new():
        """Factory function to create hipExternalMemoryHandleDesc_st objects with
        newly allocated chip.hipExternalMemoryHandleDesc_st"""
        cdef chip.hipExternalMemoryHandleDesc_st *_ptr = <chip.hipExternalMemoryHandleDesc_st *>stdlib.malloc(sizeof(chip.hipExternalMemoryHandleDesc_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalMemoryHandleDesc_st.from_ptr(_ptr, owner=True)



cdef class hipExternalMemoryBufferDesc_st:
    cdef chip.hipExternalMemoryBufferDesc_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalMemoryBufferDesc_st from_ptr(chip.hipExternalMemoryBufferDesc_st *_ptr, bint owner=False):
        """Factory function to create ``hipExternalMemoryBufferDesc_st`` objects from
        given ``chip.hipExternalMemoryBufferDesc_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalMemoryBufferDesc_st wrapper = hipExternalMemoryBufferDesc_st.__new__(hipExternalMemoryBufferDesc_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalMemoryBufferDesc_st new():
        """Factory function to create hipExternalMemoryBufferDesc_st objects with
        newly allocated chip.hipExternalMemoryBufferDesc_st"""
        cdef chip.hipExternalMemoryBufferDesc_st *_ptr = <chip.hipExternalMemoryBufferDesc_st *>stdlib.malloc(sizeof(chip.hipExternalMemoryBufferDesc_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalMemoryBufferDesc_st.from_ptr(_ptr, owner=True)


class hipExternalSemaphoreHandleType_enum(enum.IntEnum):
    hipExternalSemaphoreHandleTypeOpaqueFd = chip.hipExternalSemaphoreHandleTypeOpaqueFd
    hipExternalSemaphoreHandleTypeOpaqueWin32 = chip.hipExternalSemaphoreHandleTypeOpaqueWin32
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = chip.hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
    hipExternalSemaphoreHandleTypeD3D12Fence = chip.hipExternalSemaphoreHandleTypeD3D12Fence


cdef class hipExternalSemaphoreHandleDesc_st_union_0_struct_0:
    cdef chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0_struct_0 from_ptr(chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreHandleDesc_st_union_0_struct_0`` objects from
        given ``chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreHandleDesc_st_union_0_struct_0 wrapper = hipExternalSemaphoreHandleDesc_st_union_0_struct_0.__new__(hipExternalSemaphoreHandleDesc_st_union_0_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0_struct_0 new():
        """Factory function to create hipExternalSemaphoreHandleDesc_st_union_0_struct_0 objects with
        newly allocated chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0"""
        cdef chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0 *_ptr = <chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0 *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreHandleDesc_st_union_0_struct_0.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreHandleDesc_st_union_0:
    cdef chip.hipExternalSemaphoreHandleDesc_st_union_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0 from_ptr(chip.hipExternalSemaphoreHandleDesc_st_union_0 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreHandleDesc_st_union_0`` objects from
        given ``chip.hipExternalSemaphoreHandleDesc_st_union_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreHandleDesc_st_union_0 wrapper = hipExternalSemaphoreHandleDesc_st_union_0.__new__(hipExternalSemaphoreHandleDesc_st_union_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0 new():
        """Factory function to create hipExternalSemaphoreHandleDesc_st_union_0 objects with
        newly allocated chip.hipExternalSemaphoreHandleDesc_st_union_0"""
        cdef chip.hipExternalSemaphoreHandleDesc_st_union_0 *_ptr = <chip.hipExternalSemaphoreHandleDesc_st_union_0 *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreHandleDesc_st_union_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreHandleDesc_st_union_0.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreHandleDesc_st:
    cdef chip.hipExternalSemaphoreHandleDesc_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st from_ptr(chip.hipExternalSemaphoreHandleDesc_st *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreHandleDesc_st`` objects from
        given ``chip.hipExternalSemaphoreHandleDesc_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreHandleDesc_st wrapper = hipExternalSemaphoreHandleDesc_st.__new__(hipExternalSemaphoreHandleDesc_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st new():
        """Factory function to create hipExternalSemaphoreHandleDesc_st objects with
        newly allocated chip.hipExternalSemaphoreHandleDesc_st"""
        cdef chip.hipExternalSemaphoreHandleDesc_st *_ptr = <chip.hipExternalSemaphoreHandleDesc_st *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreHandleDesc_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreHandleDesc_st.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreSignalParams_st_struct_0_struct_0:
    cdef chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_0 from_ptr(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreSignalParams_st_struct_0_struct_0`` objects from
        given ``chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_0 wrapper = hipExternalSemaphoreSignalParams_st_struct_0_struct_0.__new__(hipExternalSemaphoreSignalParams_st_struct_0_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_0 new():
        """Factory function to create hipExternalSemaphoreSignalParams_st_struct_0_struct_0 objects with
        newly allocated chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0"""
        cdef chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0 *_ptr = <chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0 *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreSignalParams_st_struct_0_struct_0.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreSignalParams_st_struct_0_struct_1:
    cdef chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_1 from_ptr(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreSignalParams_st_struct_0_struct_1`` objects from
        given ``chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_1 wrapper = hipExternalSemaphoreSignalParams_st_struct_0_struct_1.__new__(hipExternalSemaphoreSignalParams_st_struct_0_struct_1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_1 new():
        """Factory function to create hipExternalSemaphoreSignalParams_st_struct_0_struct_1 objects with
        newly allocated chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1"""
        cdef chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1 *_ptr = <chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1 *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreSignalParams_st_struct_0_struct_1.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreSignalParams_st_struct_0:
    cdef chip.hipExternalSemaphoreSignalParams_st_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0 from_ptr(chip.hipExternalSemaphoreSignalParams_st_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreSignalParams_st_struct_0`` objects from
        given ``chip.hipExternalSemaphoreSignalParams_st_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreSignalParams_st_struct_0 wrapper = hipExternalSemaphoreSignalParams_st_struct_0.__new__(hipExternalSemaphoreSignalParams_st_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0 new():
        """Factory function to create hipExternalSemaphoreSignalParams_st_struct_0 objects with
        newly allocated chip.hipExternalSemaphoreSignalParams_st_struct_0"""
        cdef chip.hipExternalSemaphoreSignalParams_st_struct_0 *_ptr = <chip.hipExternalSemaphoreSignalParams_st_struct_0 *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreSignalParams_st_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreSignalParams_st_struct_0.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreSignalParams_st:
    cdef chip.hipExternalSemaphoreSignalParams_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st from_ptr(chip.hipExternalSemaphoreSignalParams_st *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreSignalParams_st`` objects from
        given ``chip.hipExternalSemaphoreSignalParams_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreSignalParams_st wrapper = hipExternalSemaphoreSignalParams_st.__new__(hipExternalSemaphoreSignalParams_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st new():
        """Factory function to create hipExternalSemaphoreSignalParams_st objects with
        newly allocated chip.hipExternalSemaphoreSignalParams_st"""
        cdef chip.hipExternalSemaphoreSignalParams_st *_ptr = <chip.hipExternalSemaphoreSignalParams_st *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreSignalParams_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreSignalParams_st.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreWaitParams_st_struct_0_struct_0:
    cdef chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_0 from_ptr(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreWaitParams_st_struct_0_struct_0`` objects from
        given ``chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_0 wrapper = hipExternalSemaphoreWaitParams_st_struct_0_struct_0.__new__(hipExternalSemaphoreWaitParams_st_struct_0_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_0 new():
        """Factory function to create hipExternalSemaphoreWaitParams_st_struct_0_struct_0 objects with
        newly allocated chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0"""
        cdef chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0 *_ptr = <chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0 *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreWaitParams_st_struct_0_struct_0.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreWaitParams_st_struct_0_struct_1:
    cdef chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_1 from_ptr(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreWaitParams_st_struct_0_struct_1`` objects from
        given ``chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_1 wrapper = hipExternalSemaphoreWaitParams_st_struct_0_struct_1.__new__(hipExternalSemaphoreWaitParams_st_struct_0_struct_1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_1 new():
        """Factory function to create hipExternalSemaphoreWaitParams_st_struct_0_struct_1 objects with
        newly allocated chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1"""
        cdef chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1 *_ptr = <chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1 *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreWaitParams_st_struct_0_struct_1.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreWaitParams_st_struct_0:
    cdef chip.hipExternalSemaphoreWaitParams_st_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0 from_ptr(chip.hipExternalSemaphoreWaitParams_st_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreWaitParams_st_struct_0`` objects from
        given ``chip.hipExternalSemaphoreWaitParams_st_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreWaitParams_st_struct_0 wrapper = hipExternalSemaphoreWaitParams_st_struct_0.__new__(hipExternalSemaphoreWaitParams_st_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0 new():
        """Factory function to create hipExternalSemaphoreWaitParams_st_struct_0 objects with
        newly allocated chip.hipExternalSemaphoreWaitParams_st_struct_0"""
        cdef chip.hipExternalSemaphoreWaitParams_st_struct_0 *_ptr = <chip.hipExternalSemaphoreWaitParams_st_struct_0 *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreWaitParams_st_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreWaitParams_st_struct_0.from_ptr(_ptr, owner=True)



cdef class hipExternalSemaphoreWaitParams_st:
    cdef chip.hipExternalSemaphoreWaitParams_st* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st from_ptr(chip.hipExternalSemaphoreWaitParams_st *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreWaitParams_st`` objects from
        given ``chip.hipExternalSemaphoreWaitParams_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreWaitParams_st wrapper = hipExternalSemaphoreWaitParams_st.__new__(hipExternalSemaphoreWaitParams_st)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st new():
        """Factory function to create hipExternalSemaphoreWaitParams_st objects with
        newly allocated chip.hipExternalSemaphoreWaitParams_st"""
        cdef chip.hipExternalSemaphoreWaitParams_st *_ptr = <chip.hipExternalSemaphoreWaitParams_st *>stdlib.malloc(sizeof(chip.hipExternalSemaphoreWaitParams_st))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipExternalSemaphoreWaitParams_st.from_ptr(_ptr, owner=True)


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


cdef class _hipGraphicsResource:
    cdef chip._hipGraphicsResource* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef _hipGraphicsResource from_ptr(chip._hipGraphicsResource *_ptr, bint owner=False):
        """Factory function to create ``_hipGraphicsResource`` objects from
        given ``chip._hipGraphicsResource`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef _hipGraphicsResource wrapper = _hipGraphicsResource.__new__(_hipGraphicsResource)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipGraphicsResource_t = _hipGraphicsResource


cdef class ihipGraph:
    cdef chip.ihipGraph* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ihipGraph from_ptr(chip.ihipGraph *_ptr, bint owner=False):
        """Factory function to create ``ihipGraph`` objects from
        given ``chip.ihipGraph`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihipGraph wrapper = ihipGraph.__new__(ihipGraph)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipGraph_t = ihipGraph


cdef class hipGraphNode:
    cdef chip.hipGraphNode* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipGraphNode from_ptr(chip.hipGraphNode *_ptr, bint owner=False):
        """Factory function to create ``hipGraphNode`` objects from
        given ``chip.hipGraphNode`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipGraphNode wrapper = hipGraphNode.__new__(hipGraphNode)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipGraphNode_t = hipGraphNode


cdef class hipGraphExec:
    cdef chip.hipGraphExec* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipGraphExec from_ptr(chip.hipGraphExec *_ptr, bint owner=False):
        """Factory function to create ``hipGraphExec`` objects from
        given ``chip.hipGraphExec`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipGraphExec wrapper = hipGraphExec.__new__(hipGraphExec)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipGraphExec_t = hipGraphExec


cdef class hipUserObject:
    cdef chip.hipUserObject* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipUserObject from_ptr(chip.hipUserObject *_ptr, bint owner=False):
        """Factory function to create ``hipUserObject`` objects from
        given ``chip.hipUserObject`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipUserObject wrapper = hipUserObject.__new__(hipUserObject)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipUserObject_t = hipUserObject

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


cdef class hipHostFn_t:
    cdef chip.hipHostFn_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipHostFn_t from_ptr(chip.hipHostFn_t *_ptr, bint owner=False):
        """Factory function to create ``hipHostFn_t`` objects from
        given ``chip.hipHostFn_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipHostFn_t wrapper = hipHostFn_t.__new__(hipHostFn_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class hipHostNodeParams:
    cdef chip.hipHostNodeParams* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipHostNodeParams from_ptr(chip.hipHostNodeParams *_ptr, bint owner=False):
        """Factory function to create ``hipHostNodeParams`` objects from
        given ``chip.hipHostNodeParams`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipHostNodeParams wrapper = hipHostNodeParams.__new__(hipHostNodeParams)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipHostNodeParams new():
        """Factory function to create hipHostNodeParams objects with
        newly allocated chip.hipHostNodeParams"""
        cdef chip.hipHostNodeParams *_ptr = <chip.hipHostNodeParams *>stdlib.malloc(sizeof(chip.hipHostNodeParams))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipHostNodeParams.from_ptr(_ptr, owner=True)



cdef class hipKernelNodeParams:
    cdef chip.hipKernelNodeParams* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipKernelNodeParams from_ptr(chip.hipKernelNodeParams *_ptr, bint owner=False):
        """Factory function to create ``hipKernelNodeParams`` objects from
        given ``chip.hipKernelNodeParams`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipKernelNodeParams wrapper = hipKernelNodeParams.__new__(hipKernelNodeParams)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipKernelNodeParams new():
        """Factory function to create hipKernelNodeParams objects with
        newly allocated chip.hipKernelNodeParams"""
        cdef chip.hipKernelNodeParams *_ptr = <chip.hipKernelNodeParams *>stdlib.malloc(sizeof(chip.hipKernelNodeParams))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipKernelNodeParams.from_ptr(_ptr, owner=True)



cdef class hipMemsetParams:
    cdef chip.hipMemsetParams* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipMemsetParams from_ptr(chip.hipMemsetParams *_ptr, bint owner=False):
        """Factory function to create ``hipMemsetParams`` objects from
        given ``chip.hipMemsetParams`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipMemsetParams wrapper = hipMemsetParams.__new__(hipMemsetParams)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipMemsetParams new():
        """Factory function to create hipMemsetParams objects with
        newly allocated chip.hipMemsetParams"""
        cdef chip.hipMemsetParams *_ptr = <chip.hipMemsetParams *>stdlib.malloc(sizeof(chip.hipMemsetParams))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipMemsetParams.from_ptr(_ptr, owner=True)


class hipKernelNodeAttrID(enum.IntEnum):
    hipKernelNodeAttributeAccessPolicyWindow = chip.hipKernelNodeAttributeAccessPolicyWindow
    hipKernelNodeAttributeCooperative = chip.hipKernelNodeAttributeCooperative

class hipAccessProperty(enum.IntEnum):
    hipAccessPropertyNormal = chip.hipAccessPropertyNormal
    hipAccessPropertyStreaming = chip.hipAccessPropertyStreaming
    hipAccessPropertyPersisting = chip.hipAccessPropertyPersisting


cdef class hipAccessPolicyWindow:
    cdef chip.hipAccessPolicyWindow* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipAccessPolicyWindow from_ptr(chip.hipAccessPolicyWindow *_ptr, bint owner=False):
        """Factory function to create ``hipAccessPolicyWindow`` objects from
        given ``chip.hipAccessPolicyWindow`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipAccessPolicyWindow wrapper = hipAccessPolicyWindow.__new__(hipAccessPolicyWindow)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipAccessPolicyWindow new():
        """Factory function to create hipAccessPolicyWindow objects with
        newly allocated chip.hipAccessPolicyWindow"""
        cdef chip.hipAccessPolicyWindow *_ptr = <chip.hipAccessPolicyWindow *>stdlib.malloc(sizeof(chip.hipAccessPolicyWindow))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipAccessPolicyWindow.from_ptr(_ptr, owner=True)



cdef class hipKernelNodeAttrValue:
    cdef chip.hipKernelNodeAttrValue* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipKernelNodeAttrValue from_ptr(chip.hipKernelNodeAttrValue *_ptr, bint owner=False):
        """Factory function to create ``hipKernelNodeAttrValue`` objects from
        given ``chip.hipKernelNodeAttrValue`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipKernelNodeAttrValue wrapper = hipKernelNodeAttrValue.__new__(hipKernelNodeAttrValue)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipKernelNodeAttrValue new():
        """Factory function to create hipKernelNodeAttrValue objects with
        newly allocated chip.hipKernelNodeAttrValue"""
        cdef chip.hipKernelNodeAttrValue *_ptr = <chip.hipKernelNodeAttrValue *>stdlib.malloc(sizeof(chip.hipKernelNodeAttrValue))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipKernelNodeAttrValue.from_ptr(_ptr, owner=True)


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


cdef class hipMemAllocationProp_struct_0:
    cdef chip.hipMemAllocationProp_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipMemAllocationProp_struct_0 from_ptr(chip.hipMemAllocationProp_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``hipMemAllocationProp_struct_0`` objects from
        given ``chip.hipMemAllocationProp_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipMemAllocationProp_struct_0 wrapper = hipMemAllocationProp_struct_0.__new__(hipMemAllocationProp_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipMemAllocationProp_struct_0 new():
        """Factory function to create hipMemAllocationProp_struct_0 objects with
        newly allocated chip.hipMemAllocationProp_struct_0"""
        cdef chip.hipMemAllocationProp_struct_0 *_ptr = <chip.hipMemAllocationProp_struct_0 *>stdlib.malloc(sizeof(chip.hipMemAllocationProp_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipMemAllocationProp_struct_0.from_ptr(_ptr, owner=True)



cdef class hipMemAllocationProp:
    cdef chip.hipMemAllocationProp* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipMemAllocationProp from_ptr(chip.hipMemAllocationProp *_ptr, bint owner=False):
        """Factory function to create ``hipMemAllocationProp`` objects from
        given ``chip.hipMemAllocationProp`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipMemAllocationProp wrapper = hipMemAllocationProp.__new__(hipMemAllocationProp)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipMemAllocationProp new():
        """Factory function to create hipMemAllocationProp objects with
        newly allocated chip.hipMemAllocationProp"""
        cdef chip.hipMemAllocationProp *_ptr = <chip.hipMemAllocationProp *>stdlib.malloc(sizeof(chip.hipMemAllocationProp))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipMemAllocationProp.from_ptr(_ptr, owner=True)



cdef class ihipMemGenericAllocationHandle:
    cdef chip.ihipMemGenericAllocationHandle* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ihipMemGenericAllocationHandle from_ptr(chip.ihipMemGenericAllocationHandle *_ptr, bint owner=False):
        """Factory function to create ``ihipMemGenericAllocationHandle`` objects from
        given ``chip.ihipMemGenericAllocationHandle`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihipMemGenericAllocationHandle wrapper = ihipMemGenericAllocationHandle.__new__(ihipMemGenericAllocationHandle)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hipMemGenericAllocationHandle_t = ihipMemGenericAllocationHandle

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


cdef class hipArrayMapInfo_union_0:
    cdef chip.hipArrayMapInfo_union_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipArrayMapInfo_union_0 from_ptr(chip.hipArrayMapInfo_union_0 *_ptr, bint owner=False):
        """Factory function to create ``hipArrayMapInfo_union_0`` objects from
        given ``chip.hipArrayMapInfo_union_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipArrayMapInfo_union_0 wrapper = hipArrayMapInfo_union_0.__new__(hipArrayMapInfo_union_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipArrayMapInfo_union_0 new():
        """Factory function to create hipArrayMapInfo_union_0 objects with
        newly allocated chip.hipArrayMapInfo_union_0"""
        cdef chip.hipArrayMapInfo_union_0 *_ptr = <chip.hipArrayMapInfo_union_0 *>stdlib.malloc(sizeof(chip.hipArrayMapInfo_union_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipArrayMapInfo_union_0.from_ptr(_ptr, owner=True)



cdef class hipArrayMapInfo_union_1_struct_0:
    cdef chip.hipArrayMapInfo_union_1_struct_0* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_0 from_ptr(chip.hipArrayMapInfo_union_1_struct_0 *_ptr, bint owner=False):
        """Factory function to create ``hipArrayMapInfo_union_1_struct_0`` objects from
        given ``chip.hipArrayMapInfo_union_1_struct_0`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipArrayMapInfo_union_1_struct_0 wrapper = hipArrayMapInfo_union_1_struct_0.__new__(hipArrayMapInfo_union_1_struct_0)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_0 new():
        """Factory function to create hipArrayMapInfo_union_1_struct_0 objects with
        newly allocated chip.hipArrayMapInfo_union_1_struct_0"""
        cdef chip.hipArrayMapInfo_union_1_struct_0 *_ptr = <chip.hipArrayMapInfo_union_1_struct_0 *>stdlib.malloc(sizeof(chip.hipArrayMapInfo_union_1_struct_0))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipArrayMapInfo_union_1_struct_0.from_ptr(_ptr, owner=True)



cdef class hipArrayMapInfo_union_1_struct_1:
    cdef chip.hipArrayMapInfo_union_1_struct_1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_1 from_ptr(chip.hipArrayMapInfo_union_1_struct_1 *_ptr, bint owner=False):
        """Factory function to create ``hipArrayMapInfo_union_1_struct_1`` objects from
        given ``chip.hipArrayMapInfo_union_1_struct_1`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipArrayMapInfo_union_1_struct_1 wrapper = hipArrayMapInfo_union_1_struct_1.__new__(hipArrayMapInfo_union_1_struct_1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_1 new():
        """Factory function to create hipArrayMapInfo_union_1_struct_1 objects with
        newly allocated chip.hipArrayMapInfo_union_1_struct_1"""
        cdef chip.hipArrayMapInfo_union_1_struct_1 *_ptr = <chip.hipArrayMapInfo_union_1_struct_1 *>stdlib.malloc(sizeof(chip.hipArrayMapInfo_union_1_struct_1))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipArrayMapInfo_union_1_struct_1.from_ptr(_ptr, owner=True)



cdef class hipArrayMapInfo_union_1:
    cdef chip.hipArrayMapInfo_union_1* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipArrayMapInfo_union_1 from_ptr(chip.hipArrayMapInfo_union_1 *_ptr, bint owner=False):
        """Factory function to create ``hipArrayMapInfo_union_1`` objects from
        given ``chip.hipArrayMapInfo_union_1`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipArrayMapInfo_union_1 wrapper = hipArrayMapInfo_union_1.__new__(hipArrayMapInfo_union_1)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipArrayMapInfo_union_1 new():
        """Factory function to create hipArrayMapInfo_union_1 objects with
        newly allocated chip.hipArrayMapInfo_union_1"""
        cdef chip.hipArrayMapInfo_union_1 *_ptr = <chip.hipArrayMapInfo_union_1 *>stdlib.malloc(sizeof(chip.hipArrayMapInfo_union_1))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipArrayMapInfo_union_1.from_ptr(_ptr, owner=True)



cdef class hipArrayMapInfo_union_2:
    cdef chip.hipArrayMapInfo_union_2* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipArrayMapInfo_union_2 from_ptr(chip.hipArrayMapInfo_union_2 *_ptr, bint owner=False):
        """Factory function to create ``hipArrayMapInfo_union_2`` objects from
        given ``chip.hipArrayMapInfo_union_2`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipArrayMapInfo_union_2 wrapper = hipArrayMapInfo_union_2.__new__(hipArrayMapInfo_union_2)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipArrayMapInfo_union_2 new():
        """Factory function to create hipArrayMapInfo_union_2 objects with
        newly allocated chip.hipArrayMapInfo_union_2"""
        cdef chip.hipArrayMapInfo_union_2 *_ptr = <chip.hipArrayMapInfo_union_2 *>stdlib.malloc(sizeof(chip.hipArrayMapInfo_union_2))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipArrayMapInfo_union_2.from_ptr(_ptr, owner=True)



cdef class hipArrayMapInfo:
    cdef chip.hipArrayMapInfo* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipArrayMapInfo from_ptr(chip.hipArrayMapInfo *_ptr, bint owner=False):
        """Factory function to create ``hipArrayMapInfo`` objects from
        given ``chip.hipArrayMapInfo`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipArrayMapInfo wrapper = hipArrayMapInfo.__new__(hipArrayMapInfo)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipArrayMapInfo new():
        """Factory function to create hipArrayMapInfo objects with
        newly allocated chip.hipArrayMapInfo"""
        cdef chip.hipArrayMapInfo *_ptr = <chip.hipArrayMapInfo *>stdlib.malloc(sizeof(chip.hipArrayMapInfo))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipArrayMapInfo.from_ptr(_ptr, owner=True)



cdef class hipStreamCallback_t:
    cdef chip.hipStreamCallback_t* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipStreamCallback_t from_ptr(chip.hipStreamCallback_t *_ptr, bint owner=False):
        """Factory function to create ``hipStreamCallback_t`` objects from
        given ``chip.hipStreamCallback_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipStreamCallback_t wrapper = hipStreamCallback_t.__new__(hipStreamCallback_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
