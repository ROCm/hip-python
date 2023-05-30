# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import os
import enum

import hip.hip
hip = hip.hip # makes hip types and routines accessible without import
                            # allows checks such as `hasattr(cuda.cuda,"hip")`

HIP_PYTHON_MOD = hip
globals()["HIP_PYTHON"] = True

def _hip_python_get_bool_environ_var(env_var, default):
    yes_vals = ("true", "1", "t", "y", "yes")
    no_vals = ("false", "0", "f", "n", "no")
    value = os.environ.get(env_var, default).lower()
    if value in yes_vals:
        return True
    elif value in no_vals:
        return False
    else:
        allowed_vals = ", ".join([f"'{a}'" for a in (list(yes_vals)+list(no_vals))])
        raise RuntimeError(f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}")

CU_TRSA_OVERRIDE_FORMAT = hip.HIP_TRSA_OVERRIDE_FORMAT
CU_TRSF_READ_AS_INTEGER = hip.HIP_TRSF_READ_AS_INTEGER
CU_TRSF_NORMALIZED_COORDINATES = hip.HIP_TRSF_NORMALIZED_COORDINATES
CU_TRSF_SRGB = hip.HIP_TRSF_SRGB
cudaTextureType1D = hip.hipTextureType1D
cudaTextureType2D = hip.hipTextureType2D
cudaTextureType3D = hip.hipTextureType3D
cudaTextureTypeCubemap = hip.hipTextureTypeCubemap
cudaTextureType1DLayered = hip.hipTextureType1DLayered
cudaTextureType2DLayered = hip.hipTextureType2DLayered
cudaTextureTypeCubemapLayered = hip.hipTextureTypeCubemapLayered
CU_LAUNCH_PARAM_BUFFER_POINTER = hip.HIP_LAUNCH_PARAM_BUFFER_POINTER
CU_LAUNCH_PARAM_BUFFER_SIZE = hip.HIP_LAUNCH_PARAM_BUFFER_SIZE
CU_LAUNCH_PARAM_END = hip.HIP_LAUNCH_PARAM_END
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = hip.hipIpcMemLazyEnablePeerAccess
cudaIpcMemLazyEnablePeerAccess = hip.hipIpcMemLazyEnablePeerAccess
CUDA_IPC_HANDLE_SIZE = hip.HIP_IPC_HANDLE_SIZE
CU_IPC_HANDLE_SIZE = hip.HIP_IPC_HANDLE_SIZE
CU_STREAM_DEFAULT = hip.hipStreamDefault
cudaStreamDefault = hip.hipStreamDefault
CU_STREAM_NON_BLOCKING = hip.hipStreamNonBlocking
cudaStreamNonBlocking = hip.hipStreamNonBlocking
CU_EVENT_DEFAULT = hip.hipEventDefault
cudaEventDefault = hip.hipEventDefault
CU_EVENT_BLOCKING_SYNC = hip.hipEventBlockingSync
cudaEventBlockingSync = hip.hipEventBlockingSync
CU_EVENT_DISABLE_TIMING = hip.hipEventDisableTiming
cudaEventDisableTiming = hip.hipEventDisableTiming
CU_EVENT_INTERPROCESS = hip.hipEventInterprocess
cudaEventInterprocess = hip.hipEventInterprocess
cudaHostAllocDefault = hip.hipHostMallocDefault
CU_MEMHOSTALLOC_PORTABLE = hip.hipHostMallocPortable
cudaHostAllocPortable = hip.hipHostMallocPortable
CU_MEMHOSTALLOC_DEVICEMAP = hip.hipHostMallocMapped
cudaHostAllocMapped = hip.hipHostMallocMapped
CU_MEMHOSTALLOC_WRITECOMBINED = hip.hipHostMallocWriteCombined
cudaHostAllocWriteCombined = hip.hipHostMallocWriteCombined
CU_MEM_ATTACH_GLOBAL = hip.hipMemAttachGlobal
cudaMemAttachGlobal = hip.hipMemAttachGlobal
CU_MEM_ATTACH_HOST = hip.hipMemAttachHost
cudaMemAttachHost = hip.hipMemAttachHost
CU_MEM_ATTACH_SINGLE = hip.hipMemAttachSingle
cudaMemAttachSingle = hip.hipMemAttachSingle
cudaHostRegisterDefault = hip.hipHostRegisterDefault
CU_MEMHOSTREGISTER_PORTABLE = hip.hipHostRegisterPortable
cudaHostRegisterPortable = hip.hipHostRegisterPortable
CU_MEMHOSTREGISTER_DEVICEMAP = hip.hipHostRegisterMapped
cudaHostRegisterMapped = hip.hipHostRegisterMapped
CU_MEMHOSTREGISTER_IOMEMORY = hip.hipHostRegisterIoMemory
cudaHostRegisterIoMemory = hip.hipHostRegisterIoMemory
CU_CTX_SCHED_AUTO = hip.hipDeviceScheduleAuto
cudaDeviceScheduleAuto = hip.hipDeviceScheduleAuto
CU_CTX_SCHED_SPIN = hip.hipDeviceScheduleSpin
cudaDeviceScheduleSpin = hip.hipDeviceScheduleSpin
CU_CTX_SCHED_YIELD = hip.hipDeviceScheduleYield
cudaDeviceScheduleYield = hip.hipDeviceScheduleYield
CU_CTX_BLOCKING_SYNC = hip.hipDeviceScheduleBlockingSync
CU_CTX_SCHED_BLOCKING_SYNC = hip.hipDeviceScheduleBlockingSync
cudaDeviceBlockingSync = hip.hipDeviceScheduleBlockingSync
cudaDeviceScheduleBlockingSync = hip.hipDeviceScheduleBlockingSync
CU_CTX_SCHED_MASK = hip.hipDeviceScheduleMask
cudaDeviceScheduleMask = hip.hipDeviceScheduleMask
CU_CTX_MAP_HOST = hip.hipDeviceMapHost
cudaDeviceMapHost = hip.hipDeviceMapHost
CU_CTX_LMEM_RESIZE_TO_MAX = hip.hipDeviceLmemResizeToMax
cudaDeviceLmemResizeToMax = hip.hipDeviceLmemResizeToMax
cudaArrayDefault = hip.hipArrayDefault
CUDA_ARRAY3D_LAYERED = hip.hipArrayLayered
cudaArrayLayered = hip.hipArrayLayered
CUDA_ARRAY3D_SURFACE_LDST = hip.hipArraySurfaceLoadStore
cudaArraySurfaceLoadStore = hip.hipArraySurfaceLoadStore
CUDA_ARRAY3D_CUBEMAP = hip.hipArrayCubemap
cudaArrayCubemap = hip.hipArrayCubemap
CUDA_ARRAY3D_TEXTURE_GATHER = hip.hipArrayTextureGather
cudaArrayTextureGather = hip.hipArrayTextureGather
CU_OCCUPANCY_DEFAULT = hip.hipOccupancyDefault
cudaOccupancyDefault = hip.hipOccupancyDefault
CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC = hip.hipCooperativeLaunchMultiDeviceNoPreSync
cudaCooperativeLaunchMultiDeviceNoPreSync = hip.hipCooperativeLaunchMultiDeviceNoPreSync
CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC = hip.hipCooperativeLaunchMultiDeviceNoPostSync
cudaCooperativeLaunchMultiDeviceNoPostSync = hip.hipCooperativeLaunchMultiDeviceNoPostSync
CU_DEVICE_CPU = hip.hipCpuDeviceId
cudaCpuDeviceId = hip.hipCpuDeviceId
CU_DEVICE_INVALID = hip.hipInvalidDeviceId
cudaInvalidDeviceId = hip.hipInvalidDeviceId
CU_STREAM_WAIT_VALUE_GEQ = hip.hipStreamWaitValueGte
CU_STREAM_WAIT_VALUE_EQ = hip.hipStreamWaitValueEq
CU_STREAM_WAIT_VALUE_AND = hip.hipStreamWaitValueAnd
CU_STREAM_WAIT_VALUE_NOR = hip.hipStreamWaitValueNor
HIP_SUCCESS = hip.chip.HIP_SUCCESS
HIP_ERROR_INVALID_VALUE = hip.chip.HIP_ERROR_INVALID_VALUE
HIP_ERROR_NOT_INITIALIZED = hip.chip.HIP_ERROR_NOT_INITIALIZED
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = hip.chip.HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
cdef class CUuuid_st(hip.hip.hipUUID_t):
    pass
CUuuid = hip.hipUUID
cudaUUID_t = hip.hipUUID
cdef class cudaDeviceProp(hip.hip.hipDeviceProp_t):
    pass

HIP_PYTHON_CUmemorytype_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemorytype_HALLUCINATE","false")

class _CUmemorytype_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemorytype_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemorytype_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemoryType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemorytype
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemorytype(hip._hipMemoryType__Base,metaclass=_CUmemorytype_EnumMeta):
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

HIP_PYTHON_CUmemorytype_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemorytype_enum_HALLUCINATE","false")

class _CUmemorytype_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemorytype_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemorytype_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemoryType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemorytype_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemorytype_enum(hip._hipMemoryType__Base,metaclass=_CUmemorytype_enum_EnumMeta):
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

HIP_PYTHON_cudaMemoryType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemoryType_HALLUCINATE","false")

class _cudaMemoryType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemoryType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemoryType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemoryType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaMemoryType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaMemoryType(hip._hipMemoryType__Base,metaclass=_cudaMemoryType_EnumMeta):
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
cdef class cudaPointerAttributes(hip.hip.hipPointerAttribute_t):
    pass

HIP_PYTHON_CUresult_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresult_HALLUCINATE","false")

class _CUresult_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresult_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresult_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipError_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUresult
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUresult(hip._hipError_t__Base,metaclass=_CUresult_EnumMeta):
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

HIP_PYTHON_cudaError_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaError_HALLUCINATE","false")

class _cudaError_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaError_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaError_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipError_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaError
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaError(hip._hipError_t__Base,metaclass=_cudaError_EnumMeta):
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

HIP_PYTHON_cudaError_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaError_enum_HALLUCINATE","false")

class _cudaError_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaError_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaError_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipError_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaError_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaError_enum(hip._hipError_t__Base,metaclass=_cudaError_enum_EnumMeta):
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

HIP_PYTHON_cudaError_t_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaError_t_HALLUCINATE","false")

class _cudaError_t_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaError_t_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaError_t_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipError_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaError_t
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaError_t(hip._hipError_t__Base,metaclass=_cudaError_t_EnumMeta):
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

HIP_PYTHON_CUdevice_attribute_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUdevice_attribute_HALLUCINATE","false")

class _CUdevice_attribute_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUdevice_attribute_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUdevice_attribute_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipDeviceAttribute_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUdevice_attribute
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUdevice_attribute(hip._hipDeviceAttribute_t__Base,metaclass=_CUdevice_attribute_EnumMeta):
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

HIP_PYTHON_CUdevice_attribute_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUdevice_attribute_enum_HALLUCINATE","false")

class _CUdevice_attribute_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUdevice_attribute_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUdevice_attribute_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipDeviceAttribute_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUdevice_attribute_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUdevice_attribute_enum(hip._hipDeviceAttribute_t__Base,metaclass=_CUdevice_attribute_enum_EnumMeta):
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

HIP_PYTHON_cudaDeviceAttr_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaDeviceAttr_HALLUCINATE","false")

class _cudaDeviceAttr_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaDeviceAttr_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaDeviceAttr_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipDeviceAttribute_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaDeviceAttr
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaDeviceAttr(hip._hipDeviceAttribute_t__Base,metaclass=_cudaDeviceAttr_EnumMeta):
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

HIP_PYTHON_CUcomputemode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUcomputemode_HALLUCINATE","false")

class _CUcomputemode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUcomputemode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUcomputemode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipComputeMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUcomputemode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUcomputemode(hip._hipComputeMode__Base,metaclass=_CUcomputemode_EnumMeta):
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

HIP_PYTHON_CUcomputemode_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUcomputemode_enum_HALLUCINATE","false")

class _CUcomputemode_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUcomputemode_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUcomputemode_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipComputeMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUcomputemode_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUcomputemode_enum(hip._hipComputeMode__Base,metaclass=_CUcomputemode_enum_EnumMeta):
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

HIP_PYTHON_cudaComputeMode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaComputeMode_HALLUCINATE","false")

class _cudaComputeMode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaComputeMode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaComputeMode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipComputeMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaComputeMode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaComputeMode(hip._hipComputeMode__Base,metaclass=_cudaComputeMode_EnumMeta):
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

HIP_PYTHON_cudaChannelFormatKind_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaChannelFormatKind_HALLUCINATE","false")

class _cudaChannelFormatKind_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaChannelFormatKind_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaChannelFormatKind_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipChannelFormatKind):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaChannelFormatKind
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaChannelFormatKind(hip._hipChannelFormatKind__Base,metaclass=_cudaChannelFormatKind_EnumMeta):
    cudaChannelFormatKindSigned = hip.chip.hipChannelFormatKindSigned
    hipChannelFormatKindSigned = hip.chip.hipChannelFormatKindSigned
    cudaChannelFormatKindUnsigned = hip.chip.hipChannelFormatKindUnsigned
    hipChannelFormatKindUnsigned = hip.chip.hipChannelFormatKindUnsigned
    cudaChannelFormatKindFloat = hip.chip.hipChannelFormatKindFloat
    hipChannelFormatKindFloat = hip.chip.hipChannelFormatKindFloat
    cudaChannelFormatKindNone = hip.chip.hipChannelFormatKindNone
    hipChannelFormatKindNone = hip.chip.hipChannelFormatKindNone
cdef class cudaChannelFormatDesc(hip.hip.hipChannelFormatDesc):
    pass

HIP_PYTHON_CUarray_format_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUarray_format_HALLUCINATE","false")

class _CUarray_format_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUarray_format_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUarray_format_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipArray_Format):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUarray_format
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUarray_format(hip._hipArray_Format__Base,metaclass=_CUarray_format_EnumMeta):
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

HIP_PYTHON_CUarray_format_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUarray_format_enum_HALLUCINATE","false")

class _CUarray_format_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUarray_format_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUarray_format_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipArray_Format):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUarray_format_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUarray_format_enum(hip._hipArray_Format__Base,metaclass=_CUarray_format_enum_EnumMeta):
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
cdef class CUDA_ARRAY_DESCRIPTOR(hip.hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY_DESCRIPTOR_st(hip.hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY_DESCRIPTOR_v1(hip.hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY_DESCRIPTOR_v1_st(hip.hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY_DESCRIPTOR_v2(hip.hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY3D_DESCRIPTOR(hip.hip.HIP_ARRAY3D_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY3D_DESCRIPTOR_st(hip.hip.HIP_ARRAY3D_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY3D_DESCRIPTOR_v2(hip.hip.HIP_ARRAY3D_DESCRIPTOR):
    pass
cdef class CUarray_st(hip.hip.hipArray):
    pass
cdef class cudaArray(hip.hip.hipArray):
    pass
cdef class CUDA_MEMCPY2D(hip.hip.hip_Memcpy2D):
    pass
cdef class CUDA_MEMCPY2D_st(hip.hip.hip_Memcpy2D):
    pass
cdef class CUDA_MEMCPY2D_v1(hip.hip.hip_Memcpy2D):
    pass
cdef class CUDA_MEMCPY2D_v1_st(hip.hip.hip_Memcpy2D):
    pass
cdef class CUDA_MEMCPY2D_v2(hip.hip.hip_Memcpy2D):
    pass
CUarray = hip.hipArray_t
cudaArray_t = hip.hipArray_t
cudaArray_const_t = hip.hipArray_const_t
cdef class CUmipmappedArray_st(hip.hip.hipMipmappedArray):
    pass
cdef class cudaMipmappedArray(hip.hip.hipMipmappedArray):
    pass
CUmipmappedArray = hip.hipMipmappedArray_t
cudaMipmappedArray_t = hip.hipMipmappedArray_t
cudaMipmappedArray_const_t = hip.hipMipmappedArray_const_t

HIP_PYTHON_cudaResourceType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaResourceType_HALLUCINATE","false")

class _cudaResourceType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaResourceType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaResourceType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipResourceType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaResourceType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaResourceType(hip._hipResourceType__Base,metaclass=_cudaResourceType_EnumMeta):
    cudaResourceTypeArray = hip.chip.hipResourceTypeArray
    hipResourceTypeArray = hip.chip.hipResourceTypeArray
    cudaResourceTypeMipmappedArray = hip.chip.hipResourceTypeMipmappedArray
    hipResourceTypeMipmappedArray = hip.chip.hipResourceTypeMipmappedArray
    cudaResourceTypeLinear = hip.chip.hipResourceTypeLinear
    hipResourceTypeLinear = hip.chip.hipResourceTypeLinear
    cudaResourceTypePitch2D = hip.chip.hipResourceTypePitch2D
    hipResourceTypePitch2D = hip.chip.hipResourceTypePitch2D

HIP_PYTHON_CUresourcetype_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresourcetype_enum_HALLUCINATE","false")

class _CUresourcetype_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresourcetype_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresourcetype_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.HIPresourcetype_enum):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUresourcetype_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUresourcetype_enum(hip._HIPresourcetype_enum__Base,metaclass=_CUresourcetype_enum_EnumMeta):
    CU_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    HIP_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    CU_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    HIP_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    CU_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D
    HIP_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D

HIP_PYTHON_CUresourcetype_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresourcetype_HALLUCINATE","false")

class _CUresourcetype_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresourcetype_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresourcetype_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.HIPresourcetype):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUresourcetype
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUresourcetype(hip._HIPresourcetype_enum__Base,metaclass=_CUresourcetype_EnumMeta):
    CU_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    HIP_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    CU_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    HIP_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    CU_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D
    HIP_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D

HIP_PYTHON_CUaddress_mode_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUaddress_mode_enum_HALLUCINATE","false")

class _CUaddress_mode_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUaddress_mode_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUaddress_mode_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.HIPaddress_mode_enum):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUaddress_mode_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUaddress_mode_enum(hip._HIPaddress_mode_enum__Base,metaclass=_CUaddress_mode_enum_EnumMeta):
    CU_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    HIP_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    CU_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    HIP_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    CU_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    HIP_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    CU_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER
    HIP_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER

HIP_PYTHON_CUaddress_mode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUaddress_mode_HALLUCINATE","false")

class _CUaddress_mode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUaddress_mode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUaddress_mode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.HIPaddress_mode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUaddress_mode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUaddress_mode(hip._HIPaddress_mode_enum__Base,metaclass=_CUaddress_mode_EnumMeta):
    CU_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    HIP_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    CU_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    HIP_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    CU_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    HIP_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    CU_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER
    HIP_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER

HIP_PYTHON_CUfilter_mode_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfilter_mode_enum_HALLUCINATE","false")

class _CUfilter_mode_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfilter_mode_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfilter_mode_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.HIPfilter_mode_enum):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUfilter_mode_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUfilter_mode_enum(hip._HIPfilter_mode_enum__Base,metaclass=_CUfilter_mode_enum_EnumMeta):
    CU_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    HIP_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    CU_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR
    HIP_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR

HIP_PYTHON_CUfilter_mode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfilter_mode_HALLUCINATE","false")

class _CUfilter_mode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfilter_mode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfilter_mode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.HIPfilter_mode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUfilter_mode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUfilter_mode(hip._HIPfilter_mode_enum__Base,metaclass=_CUfilter_mode_EnumMeta):
    CU_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    HIP_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    CU_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR
    HIP_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR
cdef class CUDA_TEXTURE_DESC_st(hip.hip.HIP_TEXTURE_DESC_st):
    pass
CUDA_TEXTURE_DESC = hip.HIP_TEXTURE_DESC
CUDA_TEXTURE_DESC_v1 = hip.HIP_TEXTURE_DESC

HIP_PYTHON_cudaResourceViewFormat_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaResourceViewFormat_HALLUCINATE","false")

class _cudaResourceViewFormat_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaResourceViewFormat_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaResourceViewFormat_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipResourceViewFormat):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaResourceViewFormat
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaResourceViewFormat(hip._hipResourceViewFormat__Base,metaclass=_cudaResourceViewFormat_EnumMeta):
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

HIP_PYTHON_CUresourceViewFormat_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresourceViewFormat_enum_HALLUCINATE","false")

class _CUresourceViewFormat_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresourceViewFormat_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresourceViewFormat_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.HIPresourceViewFormat_enum):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUresourceViewFormat_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUresourceViewFormat_enum(hip._HIPresourceViewFormat_enum__Base,metaclass=_CUresourceViewFormat_enum_EnumMeta):
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

HIP_PYTHON_CUresourceViewFormat_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresourceViewFormat_HALLUCINATE","false")

class _CUresourceViewFormat_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresourceViewFormat_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresourceViewFormat_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.HIPresourceViewFormat):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUresourceViewFormat
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUresourceViewFormat(hip._HIPresourceViewFormat_enum__Base,metaclass=_CUresourceViewFormat_EnumMeta):
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
cdef class cudaResourceDesc(hip.hip.hipResourceDesc):
    pass
cdef class CUDA_RESOURCE_DESC_st(hip.hip.HIP_RESOURCE_DESC_st):
    pass
CUDA_RESOURCE_DESC = hip.HIP_RESOURCE_DESC
CUDA_RESOURCE_DESC_v1 = hip.HIP_RESOURCE_DESC
cdef class cudaResourceViewDesc(hip.hip.hipResourceViewDesc):
    pass
cdef class CUDA_RESOURCE_VIEW_DESC_st(hip.hip.HIP_RESOURCE_VIEW_DESC_st):
    pass
CUDA_RESOURCE_VIEW_DESC = hip.HIP_RESOURCE_VIEW_DESC
CUDA_RESOURCE_VIEW_DESC_v1 = hip.HIP_RESOURCE_VIEW_DESC

HIP_PYTHON_cudaMemcpyKind_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemcpyKind_HALLUCINATE","false")

class _cudaMemcpyKind_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemcpyKind_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemcpyKind_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemcpyKind):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaMemcpyKind
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaMemcpyKind(hip._hipMemcpyKind__Base,metaclass=_cudaMemcpyKind_EnumMeta):
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
cdef class cudaPitchedPtr(hip.hip.hipPitchedPtr):
    pass
cdef class cudaExtent(hip.hip.hipExtent):
    pass
cdef class cudaPos(hip.hip.hipPos):
    pass
cdef class cudaMemcpy3DParms(hip.hip.hipMemcpy3DParms):
    pass
cdef class CUDA_MEMCPY3D(hip.hip.HIP_MEMCPY3D):
    pass
cdef class CUDA_MEMCPY3D_st(hip.hip.HIP_MEMCPY3D):
    pass
cdef class CUDA_MEMCPY3D_v1(hip.hip.HIP_MEMCPY3D):
    pass
cdef class CUDA_MEMCPY3D_v1_st(hip.hip.HIP_MEMCPY3D):
    pass
cdef class CUDA_MEMCPY3D_v2(hip.hip.HIP_MEMCPY3D):
    pass

HIP_PYTHON_CUfunction_attribute_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfunction_attribute_HALLUCINATE","false")

class _CUfunction_attribute_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfunction_attribute_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfunction_attribute_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipFunction_attribute):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUfunction_attribute
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUfunction_attribute(hip._hipFunction_attribute__Base,metaclass=_CUfunction_attribute_EnumMeta):
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

HIP_PYTHON_CUfunction_attribute_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfunction_attribute_enum_HALLUCINATE","false")

class _CUfunction_attribute_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfunction_attribute_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfunction_attribute_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipFunction_attribute):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUfunction_attribute_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUfunction_attribute_enum(hip._hipFunction_attribute__Base,metaclass=_CUfunction_attribute_enum_EnumMeta):
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

HIP_PYTHON_CUpointer_attribute_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUpointer_attribute_HALLUCINATE","false")

class _CUpointer_attribute_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUpointer_attribute_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUpointer_attribute_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipPointer_attribute):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUpointer_attribute
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUpointer_attribute(hip._hipPointer_attribute__Base,metaclass=_CUpointer_attribute_EnumMeta):
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

HIP_PYTHON_CUpointer_attribute_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUpointer_attribute_enum_HALLUCINATE","false")

class _CUpointer_attribute_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUpointer_attribute_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUpointer_attribute_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipPointer_attribute):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUpointer_attribute_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUpointer_attribute_enum(hip._hipPointer_attribute__Base,metaclass=_CUpointer_attribute_enum_EnumMeta):
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
cudaCreateChannelDesc = hip.hipCreateChannelDesc
CUtexObject = hip.hipTextureObject_t
CUtexObject_v1 = hip.hipTextureObject_t
cudaTextureObject_t = hip.hipTextureObject_t

HIP_PYTHON_cudaTextureAddressMode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaTextureAddressMode_HALLUCINATE","false")

class _cudaTextureAddressMode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaTextureAddressMode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaTextureAddressMode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipTextureAddressMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaTextureAddressMode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaTextureAddressMode(hip._hipTextureAddressMode__Base,metaclass=_cudaTextureAddressMode_EnumMeta):
    cudaAddressModeWrap = hip.chip.hipAddressModeWrap
    hipAddressModeWrap = hip.chip.hipAddressModeWrap
    cudaAddressModeClamp = hip.chip.hipAddressModeClamp
    hipAddressModeClamp = hip.chip.hipAddressModeClamp
    cudaAddressModeMirror = hip.chip.hipAddressModeMirror
    hipAddressModeMirror = hip.chip.hipAddressModeMirror
    cudaAddressModeBorder = hip.chip.hipAddressModeBorder
    hipAddressModeBorder = hip.chip.hipAddressModeBorder

HIP_PYTHON_cudaTextureFilterMode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaTextureFilterMode_HALLUCINATE","false")

class _cudaTextureFilterMode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaTextureFilterMode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaTextureFilterMode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipTextureFilterMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaTextureFilterMode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaTextureFilterMode(hip._hipTextureFilterMode__Base,metaclass=_cudaTextureFilterMode_EnumMeta):
    cudaFilterModePoint = hip.chip.hipFilterModePoint
    hipFilterModePoint = hip.chip.hipFilterModePoint
    cudaFilterModeLinear = hip.chip.hipFilterModeLinear
    hipFilterModeLinear = hip.chip.hipFilterModeLinear

HIP_PYTHON_cudaTextureReadMode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaTextureReadMode_HALLUCINATE","false")

class _cudaTextureReadMode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaTextureReadMode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaTextureReadMode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipTextureReadMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaTextureReadMode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaTextureReadMode(hip._hipTextureReadMode__Base,metaclass=_cudaTextureReadMode_EnumMeta):
    cudaReadModeElementType = hip.chip.hipReadModeElementType
    hipReadModeElementType = hip.chip.hipReadModeElementType
    cudaReadModeNormalizedFloat = hip.chip.hipReadModeNormalizedFloat
    hipReadModeNormalizedFloat = hip.chip.hipReadModeNormalizedFloat
cdef class CUtexref_st(hip.hip.textureReference):
    pass
cdef class textureReference(hip.hip.textureReference):
    pass
cdef class cudaTextureDesc(hip.hip.hipTextureDesc):
    pass
CUsurfObject = hip.hipSurfaceObject_t
CUsurfObject_v1 = hip.hipSurfaceObject_t
cudaSurfaceObject_t = hip.hipSurfaceObject_t
cdef class surfaceReference(hip.hip.surfaceReference):
    pass

HIP_PYTHON_cudaSurfaceBoundaryMode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaSurfaceBoundaryMode_HALLUCINATE","false")

class _cudaSurfaceBoundaryMode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaSurfaceBoundaryMode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaSurfaceBoundaryMode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipSurfaceBoundaryMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaSurfaceBoundaryMode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaSurfaceBoundaryMode(hip._hipSurfaceBoundaryMode__Base,metaclass=_cudaSurfaceBoundaryMode_EnumMeta):
    cudaBoundaryModeZero = hip.chip.hipBoundaryModeZero
    hipBoundaryModeZero = hip.chip.hipBoundaryModeZero
    cudaBoundaryModeTrap = hip.chip.hipBoundaryModeTrap
    hipBoundaryModeTrap = hip.chip.hipBoundaryModeTrap
    cudaBoundaryModeClamp = hip.chip.hipBoundaryModeClamp
    hipBoundaryModeClamp = hip.chip.hipBoundaryModeClamp
cdef class CUctx_st(hip.hip.ihipCtx_t):
    pass
CUcontext = hip.hipCtx_t

HIP_PYTHON_CUdevice_P2PAttribute_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUdevice_P2PAttribute_HALLUCINATE","false")

class _CUdevice_P2PAttribute_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUdevice_P2PAttribute_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUdevice_P2PAttribute_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipDeviceP2PAttr):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUdevice_P2PAttribute
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUdevice_P2PAttribute(hip._hipDeviceP2PAttr__Base,metaclass=_CUdevice_P2PAttribute_EnumMeta):
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

HIP_PYTHON_CUdevice_P2PAttribute_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUdevice_P2PAttribute_enum_HALLUCINATE","false")

class _CUdevice_P2PAttribute_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUdevice_P2PAttribute_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUdevice_P2PAttribute_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipDeviceP2PAttr):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUdevice_P2PAttribute_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUdevice_P2PAttribute_enum(hip._hipDeviceP2PAttr__Base,metaclass=_CUdevice_P2PAttribute_enum_EnumMeta):
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

HIP_PYTHON_cudaDeviceP2PAttr_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaDeviceP2PAttr_HALLUCINATE","false")

class _cudaDeviceP2PAttr_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaDeviceP2PAttr_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaDeviceP2PAttr_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipDeviceP2PAttr):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaDeviceP2PAttr
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaDeviceP2PAttr(hip._hipDeviceP2PAttr__Base,metaclass=_cudaDeviceP2PAttr_EnumMeta):
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
cdef class CUstream_st(hip.hip.ihipStream_t):
    pass
CUstream = hip.hipStream_t
cudaStream_t = hip.hipStream_t
cdef class CUipcMemHandle_st(hip.hip.hipIpcMemHandle_st):
    pass
cdef class cudaIpcMemHandle_st(hip.hip.hipIpcMemHandle_st):
    pass
CUipcMemHandle = hip.hipIpcMemHandle_t
CUipcMemHandle_v1 = hip.hipIpcMemHandle_t
cudaIpcMemHandle_t = hip.hipIpcMemHandle_t
cdef class CUipcEventHandle_st(hip.hip.hipIpcEventHandle_st):
    pass
cdef class cudaIpcEventHandle_st(hip.hip.hipIpcEventHandle_st):
    pass
CUipcEventHandle = hip.hipIpcEventHandle_t
CUipcEventHandle_v1 = hip.hipIpcEventHandle_t
cudaIpcEventHandle_t = hip.hipIpcEventHandle_t
cdef class CUmod_st(hip.hip.ihipModule_t):
    pass
CUmodule = hip.hipModule_t
cdef class CUfunc_st(hip.hip.ihipModuleSymbol_t):
    pass
CUfunction = hip.hipFunction_t
cudaFunction_t = hip.hipFunction_t
cdef class CUmemPoolHandle_st(hip.hip.ihipMemPoolHandle_t):
    pass
CUmemoryPool = hip.hipMemPool_t
cudaMemPool_t = hip.hipMemPool_t
cdef class cudaFuncAttributes(hip.hip.hipFuncAttributes):
    pass
cdef class CUevent_st(hip.hip.ihipEvent_t):
    pass
CUevent = hip.hipEvent_t
cudaEvent_t = hip.hipEvent_t

HIP_PYTHON_CUlimit_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUlimit_HALLUCINATE","false")

class _CUlimit_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUlimit_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUlimit_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipLimit_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUlimit
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUlimit(hip._hipLimit_t__Base,metaclass=_CUlimit_EnumMeta):
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

HIP_PYTHON_CUlimit_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUlimit_enum_HALLUCINATE","false")

class _CUlimit_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUlimit_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUlimit_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipLimit_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUlimit_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUlimit_enum(hip._hipLimit_t__Base,metaclass=_CUlimit_enum_EnumMeta):
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

HIP_PYTHON_cudaLimit_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaLimit_HALLUCINATE","false")

class _cudaLimit_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaLimit_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaLimit_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipLimit_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaLimit
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaLimit(hip._hipLimit_t__Base,metaclass=_cudaLimit_EnumMeta):
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

HIP_PYTHON_CUmem_advise_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmem_advise_HALLUCINATE","false")

class _CUmem_advise_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmem_advise_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmem_advise_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemoryAdvise):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmem_advise
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmem_advise(hip._hipMemoryAdvise__Base,metaclass=_CUmem_advise_EnumMeta):
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

HIP_PYTHON_CUmem_advise_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmem_advise_enum_HALLUCINATE","false")

class _CUmem_advise_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmem_advise_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmem_advise_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemoryAdvise):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmem_advise_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmem_advise_enum(hip._hipMemoryAdvise__Base,metaclass=_CUmem_advise_enum_EnumMeta):
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

HIP_PYTHON_cudaMemoryAdvise_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemoryAdvise_HALLUCINATE","false")

class _cudaMemoryAdvise_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemoryAdvise_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemoryAdvise_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemoryAdvise):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaMemoryAdvise
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaMemoryAdvise(hip._hipMemoryAdvise__Base,metaclass=_cudaMemoryAdvise_EnumMeta):
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

HIP_PYTHON_CUmem_range_attribute_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmem_range_attribute_HALLUCINATE","false")

class _CUmem_range_attribute_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmem_range_attribute_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmem_range_attribute_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemRangeAttribute):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmem_range_attribute
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmem_range_attribute(hip._hipMemRangeAttribute__Base,metaclass=_CUmem_range_attribute_EnumMeta):
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

HIP_PYTHON_CUmem_range_attribute_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmem_range_attribute_enum_HALLUCINATE","false")

class _CUmem_range_attribute_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmem_range_attribute_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmem_range_attribute_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemRangeAttribute):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmem_range_attribute_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmem_range_attribute_enum(hip._hipMemRangeAttribute__Base,metaclass=_CUmem_range_attribute_enum_EnumMeta):
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

HIP_PYTHON_cudaMemRangeAttribute_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemRangeAttribute_HALLUCINATE","false")

class _cudaMemRangeAttribute_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemRangeAttribute_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemRangeAttribute_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemRangeAttribute):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaMemRangeAttribute
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaMemRangeAttribute(hip._hipMemRangeAttribute__Base,metaclass=_cudaMemRangeAttribute_EnumMeta):
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

HIP_PYTHON_CUmemPool_attribute_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemPool_attribute_HALLUCINATE","false")

class _CUmemPool_attribute_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemPool_attribute_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemPool_attribute_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemPoolAttr):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemPool_attribute
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemPool_attribute(hip._hipMemPoolAttr__Base,metaclass=_CUmemPool_attribute_EnumMeta):
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

HIP_PYTHON_CUmemPool_attribute_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemPool_attribute_enum_HALLUCINATE","false")

class _CUmemPool_attribute_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemPool_attribute_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemPool_attribute_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemPoolAttr):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemPool_attribute_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemPool_attribute_enum(hip._hipMemPoolAttr__Base,metaclass=_CUmemPool_attribute_enum_EnumMeta):
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

HIP_PYTHON_cudaMemPoolAttr_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemPoolAttr_HALLUCINATE","false")

class _cudaMemPoolAttr_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemPoolAttr_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemPoolAttr_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemPoolAttr):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaMemPoolAttr
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaMemPoolAttr(hip._hipMemPoolAttr__Base,metaclass=_cudaMemPoolAttr_EnumMeta):
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

HIP_PYTHON_CUmemLocationType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemLocationType_HALLUCINATE","false")

class _CUmemLocationType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemLocationType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemLocationType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemLocationType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemLocationType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemLocationType(hip._hipMemLocationType__Base,metaclass=_CUmemLocationType_EnumMeta):
    CU_MEM_LOCATION_TYPE_INVALID = hip.chip.hipMemLocationTypeInvalid
    cudaMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    hipMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    CU_MEM_LOCATION_TYPE_DEVICE = hip.chip.hipMemLocationTypeDevice
    cudaMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice
    hipMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice

HIP_PYTHON_CUmemLocationType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemLocationType_enum_HALLUCINATE","false")

class _CUmemLocationType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemLocationType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemLocationType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemLocationType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemLocationType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemLocationType_enum(hip._hipMemLocationType__Base,metaclass=_CUmemLocationType_enum_EnumMeta):
    CU_MEM_LOCATION_TYPE_INVALID = hip.chip.hipMemLocationTypeInvalid
    cudaMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    hipMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    CU_MEM_LOCATION_TYPE_DEVICE = hip.chip.hipMemLocationTypeDevice
    cudaMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice
    hipMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice

HIP_PYTHON_cudaMemLocationType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemLocationType_HALLUCINATE","false")

class _cudaMemLocationType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemLocationType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemLocationType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemLocationType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaMemLocationType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaMemLocationType(hip._hipMemLocationType__Base,metaclass=_cudaMemLocationType_EnumMeta):
    CU_MEM_LOCATION_TYPE_INVALID = hip.chip.hipMemLocationTypeInvalid
    cudaMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    hipMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    CU_MEM_LOCATION_TYPE_DEVICE = hip.chip.hipMemLocationTypeDevice
    cudaMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice
    hipMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice
cdef class CUmemLocation(hip.hip.hipMemLocation):
    pass
cdef class CUmemLocation_st(hip.hip.hipMemLocation):
    pass
cdef class CUmemLocation_v1(hip.hip.hipMemLocation):
    pass
cdef class cudaMemLocation(hip.hip.hipMemLocation):
    pass

HIP_PYTHON_CUmemAccess_flags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAccess_flags_HALLUCINATE","false")

class _CUmemAccess_flags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAccess_flags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAccess_flags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAccessFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemAccess_flags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemAccess_flags(hip._hipMemAccessFlags__Base,metaclass=_CUmemAccess_flags_EnumMeta):
    CU_MEM_ACCESS_FLAGS_PROT_NONE = hip.chip.hipMemAccessFlagsProtNone
    cudaMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    hipMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    CU_MEM_ACCESS_FLAGS_PROT_READ = hip.chip.hipMemAccessFlagsProtRead
    cudaMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    hipMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hip.chip.hipMemAccessFlagsProtReadWrite
    cudaMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite
    hipMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite

HIP_PYTHON_CUmemAccess_flags_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAccess_flags_enum_HALLUCINATE","false")

class _CUmemAccess_flags_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAccess_flags_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAccess_flags_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAccessFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemAccess_flags_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemAccess_flags_enum(hip._hipMemAccessFlags__Base,metaclass=_CUmemAccess_flags_enum_EnumMeta):
    CU_MEM_ACCESS_FLAGS_PROT_NONE = hip.chip.hipMemAccessFlagsProtNone
    cudaMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    hipMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    CU_MEM_ACCESS_FLAGS_PROT_READ = hip.chip.hipMemAccessFlagsProtRead
    cudaMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    hipMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hip.chip.hipMemAccessFlagsProtReadWrite
    cudaMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite
    hipMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite

HIP_PYTHON_cudaMemAccessFlags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemAccessFlags_HALLUCINATE","false")

class _cudaMemAccessFlags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemAccessFlags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemAccessFlags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAccessFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaMemAccessFlags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaMemAccessFlags(hip._hipMemAccessFlags__Base,metaclass=_cudaMemAccessFlags_EnumMeta):
    CU_MEM_ACCESS_FLAGS_PROT_NONE = hip.chip.hipMemAccessFlagsProtNone
    cudaMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    hipMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    CU_MEM_ACCESS_FLAGS_PROT_READ = hip.chip.hipMemAccessFlagsProtRead
    cudaMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    hipMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hip.chip.hipMemAccessFlagsProtReadWrite
    cudaMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite
    hipMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite
cdef class CUmemAccessDesc(hip.hip.hipMemAccessDesc):
    pass
cdef class CUmemAccessDesc_st(hip.hip.hipMemAccessDesc):
    pass
cdef class CUmemAccessDesc_v1(hip.hip.hipMemAccessDesc):
    pass
cdef class cudaMemAccessDesc(hip.hip.hipMemAccessDesc):
    pass

HIP_PYTHON_CUmemAllocationType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationType_HALLUCINATE","false")

class _CUmemAllocationType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAllocationType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemAllocationType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemAllocationType(hip._hipMemAllocationType__Base,metaclass=_CUmemAllocationType_EnumMeta):
    CU_MEM_ALLOCATION_TYPE_INVALID = hip.chip.hipMemAllocationTypeInvalid
    cudaMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    hipMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    CU_MEM_ALLOCATION_TYPE_PINNED = hip.chip.hipMemAllocationTypePinned
    cudaMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    hipMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    CU_MEM_ALLOCATION_TYPE_MAX = hip.chip.hipMemAllocationTypeMax
    cudaMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax
    hipMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax

HIP_PYTHON_CUmemAllocationType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationType_enum_HALLUCINATE","false")

class _CUmemAllocationType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAllocationType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemAllocationType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemAllocationType_enum(hip._hipMemAllocationType__Base,metaclass=_CUmemAllocationType_enum_EnumMeta):
    CU_MEM_ALLOCATION_TYPE_INVALID = hip.chip.hipMemAllocationTypeInvalid
    cudaMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    hipMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    CU_MEM_ALLOCATION_TYPE_PINNED = hip.chip.hipMemAllocationTypePinned
    cudaMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    hipMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    CU_MEM_ALLOCATION_TYPE_MAX = hip.chip.hipMemAllocationTypeMax
    cudaMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax
    hipMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax

HIP_PYTHON_cudaMemAllocationType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemAllocationType_HALLUCINATE","false")

class _cudaMemAllocationType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemAllocationType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemAllocationType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAllocationType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaMemAllocationType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaMemAllocationType(hip._hipMemAllocationType__Base,metaclass=_cudaMemAllocationType_EnumMeta):
    CU_MEM_ALLOCATION_TYPE_INVALID = hip.chip.hipMemAllocationTypeInvalid
    cudaMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    hipMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    CU_MEM_ALLOCATION_TYPE_PINNED = hip.chip.hipMemAllocationTypePinned
    cudaMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    hipMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    CU_MEM_ALLOCATION_TYPE_MAX = hip.chip.hipMemAllocationTypeMax
    cudaMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax
    hipMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax

HIP_PYTHON_CUmemAllocationHandleType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationHandleType_HALLUCINATE","false")

class _CUmemAllocationHandleType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationHandleType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationHandleType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAllocationHandleType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemAllocationHandleType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemAllocationHandleType(hip._hipMemAllocationHandleType__Base,metaclass=_CUmemAllocationHandleType_EnumMeta):
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

HIP_PYTHON_CUmemAllocationHandleType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationHandleType_enum_HALLUCINATE","false")

class _CUmemAllocationHandleType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationHandleType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationHandleType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAllocationHandleType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemAllocationHandleType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemAllocationHandleType_enum(hip._hipMemAllocationHandleType__Base,metaclass=_CUmemAllocationHandleType_enum_EnumMeta):
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

HIP_PYTHON_cudaMemAllocationHandleType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemAllocationHandleType_HALLUCINATE","false")

class _cudaMemAllocationHandleType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemAllocationHandleType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemAllocationHandleType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAllocationHandleType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaMemAllocationHandleType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaMemAllocationHandleType(hip._hipMemAllocationHandleType__Base,metaclass=_cudaMemAllocationHandleType_EnumMeta):
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
cdef class CUmemPoolProps(hip.hip.hipMemPoolProps):
    pass
cdef class CUmemPoolProps_st(hip.hip.hipMemPoolProps):
    pass
cdef class CUmemPoolProps_v1(hip.hip.hipMemPoolProps):
    pass
cdef class cudaMemPoolProps(hip.hip.hipMemPoolProps):
    pass
cdef class CUmemPoolPtrExportData(hip.hip.hipMemPoolPtrExportData):
    pass
cdef class CUmemPoolPtrExportData_st(hip.hip.hipMemPoolPtrExportData):
    pass
cdef class CUmemPoolPtrExportData_v1(hip.hip.hipMemPoolPtrExportData):
    pass
cdef class cudaMemPoolPtrExportData(hip.hip.hipMemPoolPtrExportData):
    pass

HIP_PYTHON_CUjit_option_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUjit_option_HALLUCINATE","false")

class _CUjit_option_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUjit_option_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUjit_option_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipJitOption):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUjit_option
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUjit_option(hip._hipJitOption__Base,metaclass=_CUjit_option_EnumMeta):
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

HIP_PYTHON_CUjit_option_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUjit_option_enum_HALLUCINATE","false")

class _CUjit_option_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUjit_option_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUjit_option_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipJitOption):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUjit_option_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUjit_option_enum(hip._hipJitOption__Base,metaclass=_CUjit_option_enum_EnumMeta):
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

HIP_PYTHON_cudaFuncAttribute_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaFuncAttribute_HALLUCINATE","false")

class _cudaFuncAttribute_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaFuncAttribute_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaFuncAttribute_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipFuncAttribute):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaFuncAttribute
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaFuncAttribute(hip._hipFuncAttribute__Base,metaclass=_cudaFuncAttribute_EnumMeta):
    cudaFuncAttributeMaxDynamicSharedMemorySize = hip.chip.hipFuncAttributeMaxDynamicSharedMemorySize
    hipFuncAttributeMaxDynamicSharedMemorySize = hip.chip.hipFuncAttributeMaxDynamicSharedMemorySize
    cudaFuncAttributePreferredSharedMemoryCarveout = hip.chip.hipFuncAttributePreferredSharedMemoryCarveout
    hipFuncAttributePreferredSharedMemoryCarveout = hip.chip.hipFuncAttributePreferredSharedMemoryCarveout
    cudaFuncAttributeMax = hip.chip.hipFuncAttributeMax
    hipFuncAttributeMax = hip.chip.hipFuncAttributeMax

HIP_PYTHON_CUfunc_cache_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfunc_cache_HALLUCINATE","false")

class _CUfunc_cache_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfunc_cache_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfunc_cache_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipFuncCache_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUfunc_cache
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUfunc_cache(hip._hipFuncCache_t__Base,metaclass=_CUfunc_cache_EnumMeta):
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

HIP_PYTHON_CUfunc_cache_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfunc_cache_enum_HALLUCINATE","false")

class _CUfunc_cache_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfunc_cache_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfunc_cache_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipFuncCache_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUfunc_cache_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUfunc_cache_enum(hip._hipFuncCache_t__Base,metaclass=_CUfunc_cache_enum_EnumMeta):
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

HIP_PYTHON_cudaFuncCache_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaFuncCache_HALLUCINATE","false")

class _cudaFuncCache_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaFuncCache_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaFuncCache_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipFuncCache_t):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaFuncCache
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaFuncCache(hip._hipFuncCache_t__Base,metaclass=_cudaFuncCache_EnumMeta):
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

HIP_PYTHON_CUsharedconfig_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUsharedconfig_HALLUCINATE","false")

class _CUsharedconfig_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUsharedconfig_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUsharedconfig_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipSharedMemConfig):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUsharedconfig
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUsharedconfig(hip._hipSharedMemConfig__Base,metaclass=_CUsharedconfig_EnumMeta):
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = hip.chip.hipSharedMemBankSizeDefault
    cudaSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    hipSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeFourByte
    cudaSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    hipSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeEightByte
    cudaSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
    hipSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte

HIP_PYTHON_CUsharedconfig_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUsharedconfig_enum_HALLUCINATE","false")

class _CUsharedconfig_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUsharedconfig_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUsharedconfig_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipSharedMemConfig):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUsharedconfig_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUsharedconfig_enum(hip._hipSharedMemConfig__Base,metaclass=_CUsharedconfig_enum_EnumMeta):
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = hip.chip.hipSharedMemBankSizeDefault
    cudaSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    hipSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeFourByte
    cudaSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    hipSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeEightByte
    cudaSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
    hipSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte

HIP_PYTHON_cudaSharedMemConfig_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaSharedMemConfig_HALLUCINATE","false")

class _cudaSharedMemConfig_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaSharedMemConfig_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaSharedMemConfig_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipSharedMemConfig):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaSharedMemConfig
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaSharedMemConfig(hip._hipSharedMemConfig__Base,metaclass=_cudaSharedMemConfig_EnumMeta):
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = hip.chip.hipSharedMemBankSizeDefault
    cudaSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    hipSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeFourByte
    cudaSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    hipSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeEightByte
    cudaSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
    hipSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
cudaLaunchParams = hip.hipLaunchParams

HIP_PYTHON_CUexternalMemoryHandleType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUexternalMemoryHandleType_enum_HALLUCINATE","false")

class _CUexternalMemoryHandleType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUexternalMemoryHandleType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUexternalMemoryHandleType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipExternalMemoryHandleType_enum):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUexternalMemoryHandleType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUexternalMemoryHandleType_enum(hip._hipExternalMemoryHandleType_enum__Base,metaclass=_CUexternalMemoryHandleType_enum_EnumMeta):
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

HIP_PYTHON_CUexternalMemoryHandleType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUexternalMemoryHandleType_HALLUCINATE","false")

class _CUexternalMemoryHandleType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUexternalMemoryHandleType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUexternalMemoryHandleType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipExternalMemoryHandleType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUexternalMemoryHandleType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUexternalMemoryHandleType(hip._hipExternalMemoryHandleType_enum__Base,metaclass=_CUexternalMemoryHandleType_EnumMeta):
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

HIP_PYTHON_cudaExternalMemoryHandleType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaExternalMemoryHandleType_HALLUCINATE","false")

class _cudaExternalMemoryHandleType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaExternalMemoryHandleType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaExternalMemoryHandleType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipExternalMemoryHandleType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaExternalMemoryHandleType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaExternalMemoryHandleType(hip._hipExternalMemoryHandleType_enum__Base,metaclass=_cudaExternalMemoryHandleType_EnumMeta):
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
cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st(hip.hip.hipExternalMemoryHandleDesc_st):
    pass
CUDA_EXTERNAL_MEMORY_HANDLE_DESC = hip.hipExternalMemoryHandleDesc
CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = hip.hipExternalMemoryHandleDesc
cudaExternalMemoryHandleDesc = hip.hipExternalMemoryHandleDesc
cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(hip.hip.hipExternalMemoryBufferDesc_st):
    pass
CUDA_EXTERNAL_MEMORY_BUFFER_DESC = hip.hipExternalMemoryBufferDesc
CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = hip.hipExternalMemoryBufferDesc
cudaExternalMemoryBufferDesc = hip.hipExternalMemoryBufferDesc

HIP_PYTHON_CUexternalSemaphoreHandleType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUexternalSemaphoreHandleType_enum_HALLUCINATE","false")

class _CUexternalSemaphoreHandleType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUexternalSemaphoreHandleType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUexternalSemaphoreHandleType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipExternalSemaphoreHandleType_enum):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUexternalSemaphoreHandleType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUexternalSemaphoreHandleType_enum(hip._hipExternalSemaphoreHandleType_enum__Base,metaclass=_CUexternalSemaphoreHandleType_enum_EnumMeta):
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

HIP_PYTHON_CUexternalSemaphoreHandleType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUexternalSemaphoreHandleType_HALLUCINATE","false")

class _CUexternalSemaphoreHandleType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUexternalSemaphoreHandleType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUexternalSemaphoreHandleType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipExternalSemaphoreHandleType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUexternalSemaphoreHandleType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUexternalSemaphoreHandleType(hip._hipExternalSemaphoreHandleType_enum__Base,metaclass=_CUexternalSemaphoreHandleType_EnumMeta):
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

HIP_PYTHON_cudaExternalSemaphoreHandleType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaExternalSemaphoreHandleType_HALLUCINATE","false")

class _cudaExternalSemaphoreHandleType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaExternalSemaphoreHandleType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaExternalSemaphoreHandleType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipExternalSemaphoreHandleType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaExternalSemaphoreHandleType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaExternalSemaphoreHandleType(hip._hipExternalSemaphoreHandleType_enum__Base,metaclass=_cudaExternalSemaphoreHandleType_EnumMeta):
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
cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st(hip.hip.hipExternalSemaphoreHandleDesc_st):
    pass
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = hip.hipExternalSemaphoreHandleDesc
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = hip.hipExternalSemaphoreHandleDesc
cudaExternalSemaphoreHandleDesc = hip.hipExternalSemaphoreHandleDesc
cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(hip.hip.hipExternalSemaphoreSignalParams_st):
    pass
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = hip.hipExternalSemaphoreSignalParams
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = hip.hipExternalSemaphoreSignalParams
cudaExternalSemaphoreSignalParams = hip.hipExternalSemaphoreSignalParams
cudaExternalSemaphoreSignalParams_v1 = hip.hipExternalSemaphoreSignalParams
cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(hip.hip.hipExternalSemaphoreWaitParams_st):
    pass
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = hip.hipExternalSemaphoreWaitParams
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = hip.hipExternalSemaphoreWaitParams
cudaExternalSemaphoreWaitParams = hip.hipExternalSemaphoreWaitParams
cudaExternalSemaphoreWaitParams_v1 = hip.hipExternalSemaphoreWaitParams

HIP_PYTHON_CUGLDeviceList_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUGLDeviceList_HALLUCINATE","false")

class _CUGLDeviceList_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUGLDeviceList_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUGLDeviceList_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGLDeviceList):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUGLDeviceList
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUGLDeviceList(hip._hipGLDeviceList__Base,metaclass=_CUGLDeviceList_EnumMeta):
    CU_GL_DEVICE_LIST_ALL = hip.chip.hipGLDeviceListAll
    cudaGLDeviceListAll = hip.chip.hipGLDeviceListAll
    hipGLDeviceListAll = hip.chip.hipGLDeviceListAll
    CU_GL_DEVICE_LIST_CURRENT_FRAME = hip.chip.hipGLDeviceListCurrentFrame
    cudaGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    hipGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    CU_GL_DEVICE_LIST_NEXT_FRAME = hip.chip.hipGLDeviceListNextFrame
    cudaGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame
    hipGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame

HIP_PYTHON_CUGLDeviceList_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUGLDeviceList_enum_HALLUCINATE","false")

class _CUGLDeviceList_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUGLDeviceList_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUGLDeviceList_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGLDeviceList):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUGLDeviceList_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUGLDeviceList_enum(hip._hipGLDeviceList__Base,metaclass=_CUGLDeviceList_enum_EnumMeta):
    CU_GL_DEVICE_LIST_ALL = hip.chip.hipGLDeviceListAll
    cudaGLDeviceListAll = hip.chip.hipGLDeviceListAll
    hipGLDeviceListAll = hip.chip.hipGLDeviceListAll
    CU_GL_DEVICE_LIST_CURRENT_FRAME = hip.chip.hipGLDeviceListCurrentFrame
    cudaGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    hipGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    CU_GL_DEVICE_LIST_NEXT_FRAME = hip.chip.hipGLDeviceListNextFrame
    cudaGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame
    hipGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame

HIP_PYTHON_cudaGLDeviceList_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGLDeviceList_HALLUCINATE","false")

class _cudaGLDeviceList_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGLDeviceList_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGLDeviceList_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGLDeviceList):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaGLDeviceList
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaGLDeviceList(hip._hipGLDeviceList__Base,metaclass=_cudaGLDeviceList_EnumMeta):
    CU_GL_DEVICE_LIST_ALL = hip.chip.hipGLDeviceListAll
    cudaGLDeviceListAll = hip.chip.hipGLDeviceListAll
    hipGLDeviceListAll = hip.chip.hipGLDeviceListAll
    CU_GL_DEVICE_LIST_CURRENT_FRAME = hip.chip.hipGLDeviceListCurrentFrame
    cudaGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    hipGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    CU_GL_DEVICE_LIST_NEXT_FRAME = hip.chip.hipGLDeviceListNextFrame
    cudaGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame
    hipGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame

HIP_PYTHON_CUgraphicsRegisterFlags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphicsRegisterFlags_HALLUCINATE","false")

class _CUgraphicsRegisterFlags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphicsRegisterFlags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphicsRegisterFlags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphicsRegisterFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphicsRegisterFlags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphicsRegisterFlags(hip._hipGraphicsRegisterFlags__Base,metaclass=_CUgraphicsRegisterFlags_EnumMeta):
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

HIP_PYTHON_CUgraphicsRegisterFlags_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphicsRegisterFlags_enum_HALLUCINATE","false")

class _CUgraphicsRegisterFlags_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphicsRegisterFlags_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphicsRegisterFlags_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphicsRegisterFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphicsRegisterFlags_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphicsRegisterFlags_enum(hip._hipGraphicsRegisterFlags__Base,metaclass=_CUgraphicsRegisterFlags_enum_EnumMeta):
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

HIP_PYTHON_cudaGraphicsRegisterFlags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphicsRegisterFlags_HALLUCINATE","false")

class _cudaGraphicsRegisterFlags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphicsRegisterFlags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphicsRegisterFlags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphicsRegisterFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaGraphicsRegisterFlags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaGraphicsRegisterFlags(hip._hipGraphicsRegisterFlags__Base,metaclass=_cudaGraphicsRegisterFlags_EnumMeta):
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
CUgraphicsResource_st = hip.hipGraphicsResource
cudaGraphicsResource = hip.hipGraphicsResource
CUgraphicsResource = hip.hipGraphicsResource_t
cudaGraphicsResource_t = hip.hipGraphicsResource_t
cdef class CUgraph_st(hip.hip.ihipGraph):
    pass
CUgraph = hip.hipGraph_t
cudaGraph_t = hip.hipGraph_t
cdef class CUgraphNode_st(hip.hip.hipGraphNode):
    pass
CUgraphNode = hip.hipGraphNode_t
cudaGraphNode_t = hip.hipGraphNode_t
cdef class CUgraphExec_st(hip.hip.hipGraphExec):
    pass
CUgraphExec = hip.hipGraphExec_t
cudaGraphExec_t = hip.hipGraphExec_t
cdef class CUuserObject_st(hip.hip.hipUserObject):
    pass
CUuserObject = hip.hipUserObject_t
cudaUserObject_t = hip.hipUserObject_t

HIP_PYTHON_CUgraphNodeType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphNodeType_HALLUCINATE","false")

class _CUgraphNodeType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphNodeType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphNodeType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphNodeType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphNodeType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphNodeType(hip._hipGraphNodeType__Base,metaclass=_CUgraphNodeType_EnumMeta):
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

HIP_PYTHON_CUgraphNodeType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphNodeType_enum_HALLUCINATE","false")

class _CUgraphNodeType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphNodeType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphNodeType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphNodeType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphNodeType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphNodeType_enum(hip._hipGraphNodeType__Base,metaclass=_CUgraphNodeType_enum_EnumMeta):
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

HIP_PYTHON_cudaGraphNodeType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphNodeType_HALLUCINATE","false")

class _cudaGraphNodeType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphNodeType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphNodeType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphNodeType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaGraphNodeType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaGraphNodeType(hip._hipGraphNodeType__Base,metaclass=_cudaGraphNodeType_EnumMeta):
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
cdef class CUhostFn(hip.hip.hipHostFn_t):
    pass
cdef class cudaHostFn_t(hip.hip.hipHostFn_t):
    pass
cdef class CUDA_HOST_NODE_PARAMS(hip.hip.hipHostNodeParams):
    pass
cdef class CUDA_HOST_NODE_PARAMS_st(hip.hip.hipHostNodeParams):
    pass
cdef class CUDA_HOST_NODE_PARAMS_v1(hip.hip.hipHostNodeParams):
    pass
cdef class cudaHostNodeParams(hip.hip.hipHostNodeParams):
    pass
cdef class CUDA_KERNEL_NODE_PARAMS(hip.hip.hipKernelNodeParams):
    pass
cdef class CUDA_KERNEL_NODE_PARAMS_st(hip.hip.hipKernelNodeParams):
    pass
cdef class CUDA_KERNEL_NODE_PARAMS_v1(hip.hip.hipKernelNodeParams):
    pass
cdef class cudaKernelNodeParams(hip.hip.hipKernelNodeParams):
    pass
cdef class CUDA_MEMSET_NODE_PARAMS(hip.hip.hipMemsetParams):
    pass
cdef class CUDA_MEMSET_NODE_PARAMS_st(hip.hip.hipMemsetParams):
    pass
cdef class CUDA_MEMSET_NODE_PARAMS_v1(hip.hip.hipMemsetParams):
    pass
cdef class cudaMemsetParams(hip.hip.hipMemsetParams):
    pass

HIP_PYTHON_CUkernelNodeAttrID_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUkernelNodeAttrID_HALLUCINATE","false")

class _CUkernelNodeAttrID_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUkernelNodeAttrID_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUkernelNodeAttrID_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipKernelNodeAttrID):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUkernelNodeAttrID
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUkernelNodeAttrID(hip._hipKernelNodeAttrID__Base,metaclass=_CUkernelNodeAttrID_EnumMeta):
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    cudaKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    hipKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = hip.chip.hipKernelNodeAttributeCooperative
    cudaKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative
    hipKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative

HIP_PYTHON_CUkernelNodeAttrID_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUkernelNodeAttrID_enum_HALLUCINATE","false")

class _CUkernelNodeAttrID_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUkernelNodeAttrID_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUkernelNodeAttrID_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipKernelNodeAttrID):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUkernelNodeAttrID_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUkernelNodeAttrID_enum(hip._hipKernelNodeAttrID__Base,metaclass=_CUkernelNodeAttrID_enum_EnumMeta):
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    cudaKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    hipKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = hip.chip.hipKernelNodeAttributeCooperative
    cudaKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative
    hipKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative

HIP_PYTHON_cudaKernelNodeAttrID_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaKernelNodeAttrID_HALLUCINATE","false")

class _cudaKernelNodeAttrID_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaKernelNodeAttrID_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaKernelNodeAttrID_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipKernelNodeAttrID):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaKernelNodeAttrID
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaKernelNodeAttrID(hip._hipKernelNodeAttrID__Base,metaclass=_cudaKernelNodeAttrID_EnumMeta):
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    cudaKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    hipKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = hip.chip.hipKernelNodeAttributeCooperative
    cudaKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative
    hipKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative

HIP_PYTHON_CUaccessProperty_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUaccessProperty_HALLUCINATE","false")

class _CUaccessProperty_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUaccessProperty_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUaccessProperty_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipAccessProperty):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUaccessProperty
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUaccessProperty(hip._hipAccessProperty__Base,metaclass=_CUaccessProperty_EnumMeta):
    CU_ACCESS_PROPERTY_NORMAL = hip.chip.hipAccessPropertyNormal
    cudaAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    hipAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    CU_ACCESS_PROPERTY_STREAMING = hip.chip.hipAccessPropertyStreaming
    cudaAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    hipAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    CU_ACCESS_PROPERTY_PERSISTING = hip.chip.hipAccessPropertyPersisting
    cudaAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting
    hipAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting

HIP_PYTHON_CUaccessProperty_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUaccessProperty_enum_HALLUCINATE","false")

class _CUaccessProperty_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUaccessProperty_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUaccessProperty_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipAccessProperty):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUaccessProperty_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUaccessProperty_enum(hip._hipAccessProperty__Base,metaclass=_CUaccessProperty_enum_EnumMeta):
    CU_ACCESS_PROPERTY_NORMAL = hip.chip.hipAccessPropertyNormal
    cudaAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    hipAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    CU_ACCESS_PROPERTY_STREAMING = hip.chip.hipAccessPropertyStreaming
    cudaAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    hipAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    CU_ACCESS_PROPERTY_PERSISTING = hip.chip.hipAccessPropertyPersisting
    cudaAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting
    hipAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting

HIP_PYTHON_cudaAccessProperty_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaAccessProperty_HALLUCINATE","false")

class _cudaAccessProperty_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaAccessProperty_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaAccessProperty_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipAccessProperty):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaAccessProperty
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaAccessProperty(hip._hipAccessProperty__Base,metaclass=_cudaAccessProperty_EnumMeta):
    CU_ACCESS_PROPERTY_NORMAL = hip.chip.hipAccessPropertyNormal
    cudaAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    hipAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    CU_ACCESS_PROPERTY_STREAMING = hip.chip.hipAccessPropertyStreaming
    cudaAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    hipAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    CU_ACCESS_PROPERTY_PERSISTING = hip.chip.hipAccessPropertyPersisting
    cudaAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting
    hipAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting
cdef class CUaccessPolicyWindow(hip.hip.hipAccessPolicyWindow):
    pass
cdef class CUaccessPolicyWindow_st(hip.hip.hipAccessPolicyWindow):
    pass
cdef class cudaAccessPolicyWindow(hip.hip.hipAccessPolicyWindow):
    pass
cdef class CUkernelNodeAttrValue(hip.hip.hipKernelNodeAttrValue):
    pass
cdef class CUkernelNodeAttrValue_union(hip.hip.hipKernelNodeAttrValue):
    pass
cdef class CUkernelNodeAttrValue_v1(hip.hip.hipKernelNodeAttrValue):
    pass
cdef class cudaKernelNodeAttrValue(hip.hip.hipKernelNodeAttrValue):
    pass

HIP_PYTHON_CUgraphExecUpdateResult_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphExecUpdateResult_HALLUCINATE","false")

class _CUgraphExecUpdateResult_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphExecUpdateResult_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphExecUpdateResult_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphExecUpdateResult):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphExecUpdateResult
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphExecUpdateResult(hip._hipGraphExecUpdateResult__Base,metaclass=_CUgraphExecUpdateResult_EnumMeta):
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

HIP_PYTHON_CUgraphExecUpdateResult_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphExecUpdateResult_enum_HALLUCINATE","false")

class _CUgraphExecUpdateResult_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphExecUpdateResult_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphExecUpdateResult_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphExecUpdateResult):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphExecUpdateResult_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphExecUpdateResult_enum(hip._hipGraphExecUpdateResult__Base,metaclass=_CUgraphExecUpdateResult_enum_EnumMeta):
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

HIP_PYTHON_cudaGraphExecUpdateResult_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphExecUpdateResult_HALLUCINATE","false")

class _cudaGraphExecUpdateResult_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphExecUpdateResult_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphExecUpdateResult_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphExecUpdateResult):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaGraphExecUpdateResult
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaGraphExecUpdateResult(hip._hipGraphExecUpdateResult__Base,metaclass=_cudaGraphExecUpdateResult_EnumMeta):
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

HIP_PYTHON_CUstreamCaptureMode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamCaptureMode_HALLUCINATE","false")

class _CUstreamCaptureMode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamCaptureMode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamCaptureMode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipStreamCaptureMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUstreamCaptureMode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUstreamCaptureMode(hip._hipStreamCaptureMode__Base,metaclass=_CUstreamCaptureMode_EnumMeta):
    CU_STREAM_CAPTURE_MODE_GLOBAL = hip.chip.hipStreamCaptureModeGlobal
    cudaStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    hipStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = hip.chip.hipStreamCaptureModeThreadLocal
    cudaStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    hipStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    CU_STREAM_CAPTURE_MODE_RELAXED = hip.chip.hipStreamCaptureModeRelaxed
    cudaStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed
    hipStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed

HIP_PYTHON_CUstreamCaptureMode_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamCaptureMode_enum_HALLUCINATE","false")

class _CUstreamCaptureMode_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamCaptureMode_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamCaptureMode_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipStreamCaptureMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUstreamCaptureMode_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUstreamCaptureMode_enum(hip._hipStreamCaptureMode__Base,metaclass=_CUstreamCaptureMode_enum_EnumMeta):
    CU_STREAM_CAPTURE_MODE_GLOBAL = hip.chip.hipStreamCaptureModeGlobal
    cudaStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    hipStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = hip.chip.hipStreamCaptureModeThreadLocal
    cudaStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    hipStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    CU_STREAM_CAPTURE_MODE_RELAXED = hip.chip.hipStreamCaptureModeRelaxed
    cudaStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed
    hipStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed

HIP_PYTHON_cudaStreamCaptureMode_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaStreamCaptureMode_HALLUCINATE","false")

class _cudaStreamCaptureMode_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaStreamCaptureMode_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaStreamCaptureMode_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipStreamCaptureMode):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaStreamCaptureMode
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaStreamCaptureMode(hip._hipStreamCaptureMode__Base,metaclass=_cudaStreamCaptureMode_EnumMeta):
    CU_STREAM_CAPTURE_MODE_GLOBAL = hip.chip.hipStreamCaptureModeGlobal
    cudaStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    hipStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = hip.chip.hipStreamCaptureModeThreadLocal
    cudaStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    hipStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    CU_STREAM_CAPTURE_MODE_RELAXED = hip.chip.hipStreamCaptureModeRelaxed
    cudaStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed
    hipStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed

HIP_PYTHON_CUstreamCaptureStatus_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamCaptureStatus_HALLUCINATE","false")

class _CUstreamCaptureStatus_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamCaptureStatus_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamCaptureStatus_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipStreamCaptureStatus):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUstreamCaptureStatus
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUstreamCaptureStatus(hip._hipStreamCaptureStatus__Base,metaclass=_CUstreamCaptureStatus_EnumMeta):
    CU_STREAM_CAPTURE_STATUS_NONE = hip.chip.hipStreamCaptureStatusNone
    cudaStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    hipStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    CU_STREAM_CAPTURE_STATUS_ACTIVE = hip.chip.hipStreamCaptureStatusActive
    cudaStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    hipStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = hip.chip.hipStreamCaptureStatusInvalidated
    cudaStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated
    hipStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated

HIP_PYTHON_CUstreamCaptureStatus_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamCaptureStatus_enum_HALLUCINATE","false")

class _CUstreamCaptureStatus_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamCaptureStatus_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamCaptureStatus_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipStreamCaptureStatus):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUstreamCaptureStatus_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUstreamCaptureStatus_enum(hip._hipStreamCaptureStatus__Base,metaclass=_CUstreamCaptureStatus_enum_EnumMeta):
    CU_STREAM_CAPTURE_STATUS_NONE = hip.chip.hipStreamCaptureStatusNone
    cudaStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    hipStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    CU_STREAM_CAPTURE_STATUS_ACTIVE = hip.chip.hipStreamCaptureStatusActive
    cudaStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    hipStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = hip.chip.hipStreamCaptureStatusInvalidated
    cudaStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated
    hipStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated

HIP_PYTHON_cudaStreamCaptureStatus_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaStreamCaptureStatus_HALLUCINATE","false")

class _cudaStreamCaptureStatus_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaStreamCaptureStatus_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaStreamCaptureStatus_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipStreamCaptureStatus):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaStreamCaptureStatus
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaStreamCaptureStatus(hip._hipStreamCaptureStatus__Base,metaclass=_cudaStreamCaptureStatus_EnumMeta):
    CU_STREAM_CAPTURE_STATUS_NONE = hip.chip.hipStreamCaptureStatusNone
    cudaStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    hipStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    CU_STREAM_CAPTURE_STATUS_ACTIVE = hip.chip.hipStreamCaptureStatusActive
    cudaStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    hipStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = hip.chip.hipStreamCaptureStatusInvalidated
    cudaStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated
    hipStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated

HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_HALLUCINATE","false")

class _CUstreamUpdateCaptureDependencies_flags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipStreamUpdateCaptureDependenciesFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUstreamUpdateCaptureDependencies_flags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUstreamUpdateCaptureDependencies_flags(hip._hipStreamUpdateCaptureDependenciesFlags__Base,metaclass=_CUstreamUpdateCaptureDependencies_flags_EnumMeta):
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = hip.chip.hipStreamAddCaptureDependencies
    cudaStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    hipStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = hip.chip.hipStreamSetCaptureDependencies
    cudaStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies
    hipStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies

HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_enum_HALLUCINATE","false")

class _CUstreamUpdateCaptureDependencies_flags_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipStreamUpdateCaptureDependenciesFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUstreamUpdateCaptureDependencies_flags_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUstreamUpdateCaptureDependencies_flags_enum(hip._hipStreamUpdateCaptureDependenciesFlags__Base,metaclass=_CUstreamUpdateCaptureDependencies_flags_enum_EnumMeta):
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = hip.chip.hipStreamAddCaptureDependencies
    cudaStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    hipStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = hip.chip.hipStreamSetCaptureDependencies
    cudaStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies
    hipStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies

HIP_PYTHON_cudaStreamUpdateCaptureDependenciesFlags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaStreamUpdateCaptureDependenciesFlags_HALLUCINATE","false")

class _cudaStreamUpdateCaptureDependenciesFlags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaStreamUpdateCaptureDependenciesFlags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaStreamUpdateCaptureDependenciesFlags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipStreamUpdateCaptureDependenciesFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaStreamUpdateCaptureDependenciesFlags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaStreamUpdateCaptureDependenciesFlags(hip._hipStreamUpdateCaptureDependenciesFlags__Base,metaclass=_cudaStreamUpdateCaptureDependenciesFlags_EnumMeta):
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = hip.chip.hipStreamAddCaptureDependencies
    cudaStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    hipStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = hip.chip.hipStreamSetCaptureDependencies
    cudaStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies
    hipStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies

HIP_PYTHON_CUgraphMem_attribute_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphMem_attribute_HALLUCINATE","false")

class _CUgraphMem_attribute_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphMem_attribute_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphMem_attribute_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphMemAttributeType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphMem_attribute
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphMem_attribute(hip._hipGraphMemAttributeType__Base,metaclass=_CUgraphMem_attribute_EnumMeta):
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

HIP_PYTHON_CUgraphMem_attribute_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphMem_attribute_enum_HALLUCINATE","false")

class _CUgraphMem_attribute_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphMem_attribute_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphMem_attribute_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphMemAttributeType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphMem_attribute_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphMem_attribute_enum(hip._hipGraphMemAttributeType__Base,metaclass=_CUgraphMem_attribute_enum_EnumMeta):
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

HIP_PYTHON_cudaGraphMemAttributeType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphMemAttributeType_HALLUCINATE","false")

class _cudaGraphMemAttributeType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphMemAttributeType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphMemAttributeType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphMemAttributeType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaGraphMemAttributeType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaGraphMemAttributeType(hip._hipGraphMemAttributeType__Base,metaclass=_cudaGraphMemAttributeType_EnumMeta):
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

HIP_PYTHON_CUuserObject_flags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUuserObject_flags_HALLUCINATE","false")

class _CUuserObject_flags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUuserObject_flags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUuserObject_flags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipUserObjectFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUuserObject_flags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUuserObject_flags(hip._hipUserObjectFlags__Base,metaclass=_CUuserObject_flags_EnumMeta):
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = hip.chip.hipUserObjectNoDestructorSync
    cudaUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync
    hipUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync

HIP_PYTHON_CUuserObject_flags_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUuserObject_flags_enum_HALLUCINATE","false")

class _CUuserObject_flags_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUuserObject_flags_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUuserObject_flags_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipUserObjectFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUuserObject_flags_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUuserObject_flags_enum(hip._hipUserObjectFlags__Base,metaclass=_CUuserObject_flags_enum_EnumMeta):
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = hip.chip.hipUserObjectNoDestructorSync
    cudaUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync
    hipUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync

HIP_PYTHON_cudaUserObjectFlags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaUserObjectFlags_HALLUCINATE","false")

class _cudaUserObjectFlags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaUserObjectFlags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaUserObjectFlags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipUserObjectFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaUserObjectFlags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaUserObjectFlags(hip._hipUserObjectFlags__Base,metaclass=_cudaUserObjectFlags_EnumMeta):
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = hip.chip.hipUserObjectNoDestructorSync
    cudaUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync
    hipUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync

HIP_PYTHON_CUuserObjectRetain_flags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUuserObjectRetain_flags_HALLUCINATE","false")

class _CUuserObjectRetain_flags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUuserObjectRetain_flags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUuserObjectRetain_flags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipUserObjectRetainFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUuserObjectRetain_flags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUuserObjectRetain_flags(hip._hipUserObjectRetainFlags__Base,metaclass=_CUuserObjectRetain_flags_EnumMeta):
    CU_GRAPH_USER_OBJECT_MOVE = hip.chip.hipGraphUserObjectMove
    cudaGraphUserObjectMove = hip.chip.hipGraphUserObjectMove
    hipGraphUserObjectMove = hip.chip.hipGraphUserObjectMove

HIP_PYTHON_CUuserObjectRetain_flags_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUuserObjectRetain_flags_enum_HALLUCINATE","false")

class _CUuserObjectRetain_flags_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUuserObjectRetain_flags_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUuserObjectRetain_flags_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipUserObjectRetainFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUuserObjectRetain_flags_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUuserObjectRetain_flags_enum(hip._hipUserObjectRetainFlags__Base,metaclass=_CUuserObjectRetain_flags_enum_EnumMeta):
    CU_GRAPH_USER_OBJECT_MOVE = hip.chip.hipGraphUserObjectMove
    cudaGraphUserObjectMove = hip.chip.hipGraphUserObjectMove
    hipGraphUserObjectMove = hip.chip.hipGraphUserObjectMove

HIP_PYTHON_cudaUserObjectRetainFlags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaUserObjectRetainFlags_HALLUCINATE","false")

class _cudaUserObjectRetainFlags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaUserObjectRetainFlags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaUserObjectRetainFlags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipUserObjectRetainFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaUserObjectRetainFlags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaUserObjectRetainFlags(hip._hipUserObjectRetainFlags__Base,metaclass=_cudaUserObjectRetainFlags_EnumMeta):
    CU_GRAPH_USER_OBJECT_MOVE = hip.chip.hipGraphUserObjectMove
    cudaGraphUserObjectMove = hip.chip.hipGraphUserObjectMove
    hipGraphUserObjectMove = hip.chip.hipGraphUserObjectMove

HIP_PYTHON_CUgraphInstantiate_flags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphInstantiate_flags_HALLUCINATE","false")

class _CUgraphInstantiate_flags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphInstantiate_flags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphInstantiate_flags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphInstantiateFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphInstantiate_flags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphInstantiate_flags(hip._hipGraphInstantiateFlags__Base,metaclass=_CUgraphInstantiate_flags_EnumMeta):
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    cudaGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    hipGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch

HIP_PYTHON_CUgraphInstantiate_flags_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphInstantiate_flags_enum_HALLUCINATE","false")

class _CUgraphInstantiate_flags_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphInstantiate_flags_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphInstantiate_flags_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphInstantiateFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUgraphInstantiate_flags_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUgraphInstantiate_flags_enum(hip._hipGraphInstantiateFlags__Base,metaclass=_CUgraphInstantiate_flags_enum_EnumMeta):
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    cudaGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    hipGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch

HIP_PYTHON_cudaGraphInstantiateFlags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphInstantiateFlags_HALLUCINATE","false")

class _cudaGraphInstantiateFlags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphInstantiateFlags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphInstantiateFlags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipGraphInstantiateFlags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return cudaGraphInstantiateFlags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class cudaGraphInstantiateFlags(hip._hipGraphInstantiateFlags__Base,metaclass=_cudaGraphInstantiateFlags_EnumMeta):
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    cudaGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    hipGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
cdef class CUmemAllocationProp(hip.hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_st(hip.hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_v1(hip.hip.hipMemAllocationProp):
    pass
CUmemGenericAllocationHandle = hip.hipMemGenericAllocationHandle_t
CUmemGenericAllocationHandle_v1 = hip.hipMemGenericAllocationHandle_t

HIP_PYTHON_CUmemAllocationGranularity_flags_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationGranularity_flags_HALLUCINATE","false")

class _CUmemAllocationGranularity_flags_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationGranularity_flags_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationGranularity_flags_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAllocationGranularity_flags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemAllocationGranularity_flags
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemAllocationGranularity_flags(hip._hipMemAllocationGranularity_flags__Base,metaclass=_CUmemAllocationGranularity_flags_EnumMeta):
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = hip.chip.hipMemAllocationGranularityMinimum
    hipMemAllocationGranularityMinimum = hip.chip.hipMemAllocationGranularityMinimum
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = hip.chip.hipMemAllocationGranularityRecommended
    hipMemAllocationGranularityRecommended = hip.chip.hipMemAllocationGranularityRecommended

HIP_PYTHON_CUmemAllocationGranularity_flags_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationGranularity_flags_enum_HALLUCINATE","false")

class _CUmemAllocationGranularity_flags_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationGranularity_flags_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationGranularity_flags_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemAllocationGranularity_flags):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemAllocationGranularity_flags_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemAllocationGranularity_flags_enum(hip._hipMemAllocationGranularity_flags__Base,metaclass=_CUmemAllocationGranularity_flags_enum_EnumMeta):
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = hip.chip.hipMemAllocationGranularityMinimum
    hipMemAllocationGranularityMinimum = hip.chip.hipMemAllocationGranularityMinimum
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = hip.chip.hipMemAllocationGranularityRecommended
    hipMemAllocationGranularityRecommended = hip.chip.hipMemAllocationGranularityRecommended

HIP_PYTHON_CUmemHandleType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemHandleType_HALLUCINATE","false")

class _CUmemHandleType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemHandleType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemHandleType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemHandleType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemHandleType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemHandleType(hip._hipMemHandleType__Base,metaclass=_CUmemHandleType_EnumMeta):
    CU_MEM_HANDLE_TYPE_GENERIC = hip.chip.hipMemHandleTypeGeneric
    hipMemHandleTypeGeneric = hip.chip.hipMemHandleTypeGeneric

HIP_PYTHON_CUmemHandleType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemHandleType_enum_HALLUCINATE","false")

class _CUmemHandleType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemHandleType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemHandleType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemHandleType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemHandleType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemHandleType_enum(hip._hipMemHandleType__Base,metaclass=_CUmemHandleType_enum_EnumMeta):
    CU_MEM_HANDLE_TYPE_GENERIC = hip.chip.hipMemHandleTypeGeneric
    hipMemHandleTypeGeneric = hip.chip.hipMemHandleTypeGeneric

HIP_PYTHON_CUmemOperationType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemOperationType_HALLUCINATE","false")

class _CUmemOperationType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemOperationType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemOperationType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemOperationType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemOperationType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemOperationType(hip._hipMemOperationType__Base,metaclass=_CUmemOperationType_EnumMeta):
    CU_MEM_OPERATION_TYPE_MAP = hip.chip.hipMemOperationTypeMap
    hipMemOperationTypeMap = hip.chip.hipMemOperationTypeMap
    CU_MEM_OPERATION_TYPE_UNMAP = hip.chip.hipMemOperationTypeUnmap
    hipMemOperationTypeUnmap = hip.chip.hipMemOperationTypeUnmap

HIP_PYTHON_CUmemOperationType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemOperationType_enum_HALLUCINATE","false")

class _CUmemOperationType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemOperationType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemOperationType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipMemOperationType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUmemOperationType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUmemOperationType_enum(hip._hipMemOperationType__Base,metaclass=_CUmemOperationType_enum_EnumMeta):
    CU_MEM_OPERATION_TYPE_MAP = hip.chip.hipMemOperationTypeMap
    hipMemOperationTypeMap = hip.chip.hipMemOperationTypeMap
    CU_MEM_OPERATION_TYPE_UNMAP = hip.chip.hipMemOperationTypeUnmap
    hipMemOperationTypeUnmap = hip.chip.hipMemOperationTypeUnmap

HIP_PYTHON_CUarraySparseSubresourceType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUarraySparseSubresourceType_HALLUCINATE","false")

class _CUarraySparseSubresourceType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUarraySparseSubresourceType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUarraySparseSubresourceType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipArraySparseSubresourceType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUarraySparseSubresourceType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUarraySparseSubresourceType(hip._hipArraySparseSubresourceType__Base,metaclass=_CUarraySparseSubresourceType_EnumMeta):
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = hip.chip.hipArraySparseSubresourceTypeSparseLevel
    hipArraySparseSubresourceTypeSparseLevel = hip.chip.hipArraySparseSubresourceTypeSparseLevel
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = hip.chip.hipArraySparseSubresourceTypeMiptail
    hipArraySparseSubresourceTypeMiptail = hip.chip.hipArraySparseSubresourceTypeMiptail

HIP_PYTHON_CUarraySparseSubresourceType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUarraySparseSubresourceType_enum_HALLUCINATE","false")

class _CUarraySparseSubresourceType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUarraySparseSubresourceType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUarraySparseSubresourceType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hip.hipArraySparseSubresourceType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUarraySparseSubresourceType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUarraySparseSubresourceType_enum(hip._hipArraySparseSubresourceType__Base,metaclass=_CUarraySparseSubresourceType_enum_EnumMeta):
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = hip.chip.hipArraySparseSubresourceTypeSparseLevel
    hipArraySparseSubresourceTypeSparseLevel = hip.chip.hipArraySparseSubresourceTypeSparseLevel
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = hip.chip.hipArraySparseSubresourceTypeMiptail
    hipArraySparseSubresourceTypeMiptail = hip.chip.hipArraySparseSubresourceTypeMiptail
cdef class CUarrayMapInfo(hip.hip.hipArrayMapInfo):
    pass
cdef class CUarrayMapInfo_st(hip.hip.hipArrayMapInfo):
    pass
cdef class CUarrayMapInfo_v1(hip.hip.hipArrayMapInfo):
    pass
cuInit = hip.hipInit
cuDriverGetVersion = hip.hipDriverGetVersion
cudaDriverGetVersion = hip.hipDriverGetVersion
cudaRuntimeGetVersion = hip.hipRuntimeGetVersion
cuDeviceGet = hip.hipDeviceGet
cuDeviceComputeCapability = hip.hipDeviceComputeCapability
cuDeviceGetName = hip.hipDeviceGetName
cuDeviceGetUuid = hip.hipDeviceGetUuid
cuDeviceGetUuid_v2 = hip.hipDeviceGetUuid
cudaDeviceGetP2PAttribute = hip.hipDeviceGetP2PAttribute
cuDeviceGetP2PAttribute = hip.hipDeviceGetP2PAttribute
cudaDeviceGetPCIBusId = hip.hipDeviceGetPCIBusId
cuDeviceGetPCIBusId = hip.hipDeviceGetPCIBusId
cudaDeviceGetByPCIBusId = hip.hipDeviceGetByPCIBusId
cuDeviceGetByPCIBusId = hip.hipDeviceGetByPCIBusId
cuDeviceTotalMem = hip.hipDeviceTotalMem
cuDeviceTotalMem_v2 = hip.hipDeviceTotalMem
cudaDeviceSynchronize = hip.hipDeviceSynchronize
cudaThreadSynchronize = hip.hipDeviceSynchronize
cudaDeviceReset = hip.hipDeviceReset
cudaThreadExit = hip.hipDeviceReset
cudaSetDevice = hip.hipSetDevice
cudaGetDevice = hip.hipGetDevice
cuDeviceGetCount = hip.hipGetDeviceCount
cudaGetDeviceCount = hip.hipGetDeviceCount
cuDeviceGetAttribute = hip.hipDeviceGetAttribute
cudaDeviceGetAttribute = hip.hipDeviceGetAttribute
cuDeviceGetDefaultMemPool = hip.hipDeviceGetDefaultMemPool
cudaDeviceGetDefaultMemPool = hip.hipDeviceGetDefaultMemPool
cuDeviceSetMemPool = hip.hipDeviceSetMemPool
cudaDeviceSetMemPool = hip.hipDeviceSetMemPool
cuDeviceGetMemPool = hip.hipDeviceGetMemPool
cudaDeviceGetMemPool = hip.hipDeviceGetMemPool
cudaGetDeviceProperties = hip.hipGetDeviceProperties
cudaDeviceSetCacheConfig = hip.hipDeviceSetCacheConfig
cudaThreadSetCacheConfig = hip.hipDeviceSetCacheConfig
cudaDeviceGetCacheConfig = hip.hipDeviceGetCacheConfig
cudaThreadGetCacheConfig = hip.hipDeviceGetCacheConfig
cudaDeviceGetLimit = hip.hipDeviceGetLimit
cuCtxGetLimit = hip.hipDeviceGetLimit
cudaDeviceSetLimit = hip.hipDeviceSetLimit
cuCtxSetLimit = hip.hipDeviceSetLimit
cudaDeviceGetSharedMemConfig = hip.hipDeviceGetSharedMemConfig
cudaGetDeviceFlags = hip.hipGetDeviceFlags
cudaDeviceSetSharedMemConfig = hip.hipDeviceSetSharedMemConfig
cudaSetDeviceFlags = hip.hipSetDeviceFlags
cudaChooseDevice = hip.hipChooseDevice
cudaIpcGetMemHandle = hip.hipIpcGetMemHandle
cuIpcGetMemHandle = hip.hipIpcGetMemHandle
cudaIpcOpenMemHandle = hip.hipIpcOpenMemHandle
cuIpcOpenMemHandle = hip.hipIpcOpenMemHandle
cudaIpcCloseMemHandle = hip.hipIpcCloseMemHandle
cuIpcCloseMemHandle = hip.hipIpcCloseMemHandle
cudaIpcGetEventHandle = hip.hipIpcGetEventHandle
cuIpcGetEventHandle = hip.hipIpcGetEventHandle
cudaIpcOpenEventHandle = hip.hipIpcOpenEventHandle
cuIpcOpenEventHandle = hip.hipIpcOpenEventHandle
cudaFuncSetAttribute = hip.hipFuncSetAttribute
cudaFuncSetCacheConfig = hip.hipFuncSetCacheConfig
cudaFuncSetSharedMemConfig = hip.hipFuncSetSharedMemConfig
cudaGetLastError = hip.hipGetLastError
cudaPeekAtLastError = hip.hipPeekAtLastError
cudaGetErrorName = hip.hipGetErrorName
cudaGetErrorString = hip.hipGetErrorString
cuGetErrorName = hip.hipDrvGetErrorName
cuGetErrorString = hip.hipDrvGetErrorString
cudaStreamCreate = hip.hipStreamCreate
cuStreamCreate = hip.hipStreamCreateWithFlags
cudaStreamCreateWithFlags = hip.hipStreamCreateWithFlags
cuStreamCreateWithPriority = hip.hipStreamCreateWithPriority
cudaStreamCreateWithPriority = hip.hipStreamCreateWithPriority
cudaDeviceGetStreamPriorityRange = hip.hipDeviceGetStreamPriorityRange
cuCtxGetStreamPriorityRange = hip.hipDeviceGetStreamPriorityRange
cuStreamDestroy = hip.hipStreamDestroy
cuStreamDestroy_v2 = hip.hipStreamDestroy
cudaStreamDestroy = hip.hipStreamDestroy
cuStreamQuery = hip.hipStreamQuery
cudaStreamQuery = hip.hipStreamQuery
cuStreamSynchronize = hip.hipStreamSynchronize
cudaStreamSynchronize = hip.hipStreamSynchronize
cuStreamWaitEvent = hip.hipStreamWaitEvent
cudaStreamWaitEvent = hip.hipStreamWaitEvent
cuStreamGetFlags = hip.hipStreamGetFlags
cudaStreamGetFlags = hip.hipStreamGetFlags
cuStreamGetPriority = hip.hipStreamGetPriority
cudaStreamGetPriority = hip.hipStreamGetPriority
cdef class CUstreamCallback(hip.hip.hipStreamCallback_t):
    pass
cdef class cudaStreamCallback_t(hip.hip.hipStreamCallback_t):
    pass
cuStreamAddCallback = hip.hipStreamAddCallback
cudaStreamAddCallback = hip.hipStreamAddCallback
cuStreamWaitValue32 = hip.hipStreamWaitValue32
cuStreamWaitValue32_v2 = hip.hipStreamWaitValue32
cuStreamWaitValue64 = hip.hipStreamWaitValue64
cuStreamWaitValue64_v2 = hip.hipStreamWaitValue64
cuStreamWriteValue32 = hip.hipStreamWriteValue32
cuStreamWriteValue32_v2 = hip.hipStreamWriteValue32
cuStreamWriteValue64 = hip.hipStreamWriteValue64
cuStreamWriteValue64_v2 = hip.hipStreamWriteValue64
cuEventCreate = hip.hipEventCreateWithFlags
cudaEventCreateWithFlags = hip.hipEventCreateWithFlags
cudaEventCreate = hip.hipEventCreate
cuEventRecord = hip.hipEventRecord
cudaEventRecord = hip.hipEventRecord
cuEventDestroy = hip.hipEventDestroy
cuEventDestroy_v2 = hip.hipEventDestroy
cudaEventDestroy = hip.hipEventDestroy
cuEventSynchronize = hip.hipEventSynchronize
cudaEventSynchronize = hip.hipEventSynchronize
cuEventElapsedTime = hip.hipEventElapsedTime
cudaEventElapsedTime = hip.hipEventElapsedTime
cuEventQuery = hip.hipEventQuery
cudaEventQuery = hip.hipEventQuery
cudaPointerGetAttributes = hip.hipPointerGetAttributes
cuPointerGetAttribute = hip.hipPointerGetAttribute
cuPointerGetAttributes = hip.hipDrvPointerGetAttributes
cuImportExternalSemaphore = hip.hipImportExternalSemaphore
cudaImportExternalSemaphore = hip.hipImportExternalSemaphore
cuSignalExternalSemaphoresAsync = hip.hipSignalExternalSemaphoresAsync
cudaSignalExternalSemaphoresAsync = hip.hipSignalExternalSemaphoresAsync
cuWaitExternalSemaphoresAsync = hip.hipWaitExternalSemaphoresAsync
cudaWaitExternalSemaphoresAsync = hip.hipWaitExternalSemaphoresAsync
cuDestroyExternalSemaphore = hip.hipDestroyExternalSemaphore
cudaDestroyExternalSemaphore = hip.hipDestroyExternalSemaphore
cuImportExternalMemory = hip.hipImportExternalMemory
cudaImportExternalMemory = hip.hipImportExternalMemory
cuExternalMemoryGetMappedBuffer = hip.hipExternalMemoryGetMappedBuffer
cudaExternalMemoryGetMappedBuffer = hip.hipExternalMemoryGetMappedBuffer
cuDestroyExternalMemory = hip.hipDestroyExternalMemory
cudaDestroyExternalMemory = hip.hipDestroyExternalMemory
cuMemAlloc = hip.hipMalloc
cuMemAlloc_v2 = hip.hipMalloc
cudaMalloc = hip.hipMalloc
cuMemAllocHost = hip.hipMemAllocHost
cuMemAllocHost_v2 = hip.hipMemAllocHost
cudaMallocHost = hip.hipHostMalloc
cuMemAllocManaged = hip.hipMallocManaged
cudaMallocManaged = hip.hipMallocManaged
cudaMemPrefetchAsync = hip.hipMemPrefetchAsync
cuMemPrefetchAsync = hip.hipMemPrefetchAsync
cudaMemAdvise = hip.hipMemAdvise
cuMemAdvise = hip.hipMemAdvise
cudaMemRangeGetAttribute = hip.hipMemRangeGetAttribute
cuMemRangeGetAttribute = hip.hipMemRangeGetAttribute
cudaMemRangeGetAttributes = hip.hipMemRangeGetAttributes
cuMemRangeGetAttributes = hip.hipMemRangeGetAttributes
cuStreamAttachMemAsync = hip.hipStreamAttachMemAsync
cudaStreamAttachMemAsync = hip.hipStreamAttachMemAsync
cudaMallocAsync = hip.hipMallocAsync
cuMemAllocAsync = hip.hipMallocAsync
cudaFreeAsync = hip.hipFreeAsync
cuMemFreeAsync = hip.hipFreeAsync
cudaMemPoolTrimTo = hip.hipMemPoolTrimTo
cuMemPoolTrimTo = hip.hipMemPoolTrimTo
cudaMemPoolSetAttribute = hip.hipMemPoolSetAttribute
cuMemPoolSetAttribute = hip.hipMemPoolSetAttribute
cudaMemPoolGetAttribute = hip.hipMemPoolGetAttribute
cuMemPoolGetAttribute = hip.hipMemPoolGetAttribute
cudaMemPoolSetAccess = hip.hipMemPoolSetAccess
cuMemPoolSetAccess = hip.hipMemPoolSetAccess
cudaMemPoolGetAccess = hip.hipMemPoolGetAccess
cuMemPoolGetAccess = hip.hipMemPoolGetAccess
cudaMemPoolCreate = hip.hipMemPoolCreate
cuMemPoolCreate = hip.hipMemPoolCreate
cudaMemPoolDestroy = hip.hipMemPoolDestroy
cuMemPoolDestroy = hip.hipMemPoolDestroy
cudaMallocFromPoolAsync = hip.hipMallocFromPoolAsync
cuMemAllocFromPoolAsync = hip.hipMallocFromPoolAsync
cudaMemPoolExportToShareableHandle = hip.hipMemPoolExportToShareableHandle
cuMemPoolExportToShareableHandle = hip.hipMemPoolExportToShareableHandle
cudaMemPoolImportFromShareableHandle = hip.hipMemPoolImportFromShareableHandle
cuMemPoolImportFromShareableHandle = hip.hipMemPoolImportFromShareableHandle
cudaMemPoolExportPointer = hip.hipMemPoolExportPointer
cuMemPoolExportPointer = hip.hipMemPoolExportPointer
cudaMemPoolImportPointer = hip.hipMemPoolImportPointer
cuMemPoolImportPointer = hip.hipMemPoolImportPointer
cuMemHostAlloc = hip.hipHostAlloc
cudaHostAlloc = hip.hipHostAlloc
cuMemHostGetDevicePointer = hip.hipHostGetDevicePointer
cuMemHostGetDevicePointer_v2 = hip.hipHostGetDevicePointer
cudaHostGetDevicePointer = hip.hipHostGetDevicePointer
cuMemHostGetFlags = hip.hipHostGetFlags
cudaHostGetFlags = hip.hipHostGetFlags
cuMemHostRegister = hip.hipHostRegister
cuMemHostRegister_v2 = hip.hipHostRegister
cudaHostRegister = hip.hipHostRegister
cuMemHostUnregister = hip.hipHostUnregister
cudaHostUnregister = hip.hipHostUnregister
cudaMallocPitch = hip.hipMallocPitch
cuMemAllocPitch = hip.hipMemAllocPitch
cuMemAllocPitch_v2 = hip.hipMemAllocPitch
cuMemFree = hip.hipFree
cuMemFree_v2 = hip.hipFree
cudaFree = hip.hipFree
cuMemFreeHost = hip.hipHostFree
cudaFreeHost = hip.hipHostFree
cudaMemcpy = hip.hipMemcpy
cuMemcpyHtoD = hip.hipMemcpyHtoD
cuMemcpyHtoD_v2 = hip.hipMemcpyHtoD
cuMemcpyDtoH = hip.hipMemcpyDtoH
cuMemcpyDtoH_v2 = hip.hipMemcpyDtoH
cuMemcpyDtoD = hip.hipMemcpyDtoD
cuMemcpyDtoD_v2 = hip.hipMemcpyDtoD
cuMemcpyHtoDAsync = hip.hipMemcpyHtoDAsync
cuMemcpyHtoDAsync_v2 = hip.hipMemcpyHtoDAsync
cuMemcpyDtoHAsync = hip.hipMemcpyDtoHAsync
cuMemcpyDtoHAsync_v2 = hip.hipMemcpyDtoHAsync
cuMemcpyDtoDAsync = hip.hipMemcpyDtoDAsync
cuMemcpyDtoDAsync_v2 = hip.hipMemcpyDtoDAsync
cuModuleGetGlobal = hip.hipModuleGetGlobal
cuModuleGetGlobal_v2 = hip.hipModuleGetGlobal
cudaGetSymbolAddress = hip.hipGetSymbolAddress
cudaGetSymbolSize = hip.hipGetSymbolSize
cudaMemcpyToSymbol = hip.hipMemcpyToSymbol
cudaMemcpyToSymbolAsync = hip.hipMemcpyToSymbolAsync
cudaMemcpyFromSymbol = hip.hipMemcpyFromSymbol
cudaMemcpyFromSymbolAsync = hip.hipMemcpyFromSymbolAsync
cudaMemcpyAsync = hip.hipMemcpyAsync
cudaMemset = hip.hipMemset
cuMemsetD8 = hip.hipMemsetD8
cuMemsetD8_v2 = hip.hipMemsetD8
cuMemsetD8Async = hip.hipMemsetD8Async
cuMemsetD16 = hip.hipMemsetD16
cuMemsetD16_v2 = hip.hipMemsetD16
cuMemsetD16Async = hip.hipMemsetD16Async
cuMemsetD32 = hip.hipMemsetD32
cuMemsetD32_v2 = hip.hipMemsetD32
cudaMemsetAsync = hip.hipMemsetAsync
cuMemsetD32Async = hip.hipMemsetD32Async
cudaMemset2D = hip.hipMemset2D
cudaMemset2DAsync = hip.hipMemset2DAsync
cudaMemset3D = hip.hipMemset3D
cudaMemset3DAsync = hip.hipMemset3DAsync
cuMemGetInfo = hip.hipMemGetInfo
cuMemGetInfo_v2 = hip.hipMemGetInfo
cudaMemGetInfo = hip.hipMemGetInfo
cudaMallocArray = hip.hipMallocArray
cuArrayCreate = hip.hipArrayCreate
cuArrayCreate_v2 = hip.hipArrayCreate
cuArrayDestroy = hip.hipArrayDestroy
cuArray3DCreate = hip.hipArray3DCreate
cuArray3DCreate_v2 = hip.hipArray3DCreate
cudaMalloc3D = hip.hipMalloc3D
cudaFreeArray = hip.hipFreeArray
cudaFreeMipmappedArray = hip.hipFreeMipmappedArray
cudaMalloc3DArray = hip.hipMalloc3DArray
cudaMallocMipmappedArray = hip.hipMallocMipmappedArray
cudaGetMipmappedArrayLevel = hip.hipGetMipmappedArrayLevel
cudaMemcpy2D = hip.hipMemcpy2D
cuMemcpy2D = hip.hipMemcpyParam2D
cuMemcpy2D_v2 = hip.hipMemcpyParam2D
cuMemcpy2DAsync = hip.hipMemcpyParam2DAsync
cuMemcpy2DAsync_v2 = hip.hipMemcpyParam2DAsync
cudaMemcpy2DAsync = hip.hipMemcpy2DAsync
cudaMemcpy2DToArray = hip.hipMemcpy2DToArray
cudaMemcpy2DToArrayAsync = hip.hipMemcpy2DToArrayAsync
cudaMemcpyToArray = hip.hipMemcpyToArray
cudaMemcpyFromArray = hip.hipMemcpyFromArray
cudaMemcpy2DFromArray = hip.hipMemcpy2DFromArray
cudaMemcpy2DFromArrayAsync = hip.hipMemcpy2DFromArrayAsync
cuMemcpyAtoH = hip.hipMemcpyAtoH
cuMemcpyAtoH_v2 = hip.hipMemcpyAtoH
cuMemcpyHtoA = hip.hipMemcpyHtoA
cuMemcpyHtoA_v2 = hip.hipMemcpyHtoA
cudaMemcpy3D = hip.hipMemcpy3D
cudaMemcpy3DAsync = hip.hipMemcpy3DAsync
cuMemcpy3D = hip.hipDrvMemcpy3D
cuMemcpy3D_v2 = hip.hipDrvMemcpy3D
cuMemcpy3DAsync = hip.hipDrvMemcpy3DAsync
cuMemcpy3DAsync_v2 = hip.hipDrvMemcpy3DAsync
cuDeviceCanAccessPeer = hip.hipDeviceCanAccessPeer
cudaDeviceCanAccessPeer = hip.hipDeviceCanAccessPeer
cudaDeviceEnablePeerAccess = hip.hipDeviceEnablePeerAccess
cudaDeviceDisablePeerAccess = hip.hipDeviceDisablePeerAccess
cuMemGetAddressRange = hip.hipMemGetAddressRange
cuMemGetAddressRange_v2 = hip.hipMemGetAddressRange
cudaMemcpyPeer = hip.hipMemcpyPeer
cudaMemcpyPeerAsync = hip.hipMemcpyPeerAsync
cuCtxCreate = hip.hipCtxCreate
cuCtxCreate_v2 = hip.hipCtxCreate
cuCtxDestroy = hip.hipCtxDestroy
cuCtxDestroy_v2 = hip.hipCtxDestroy
cuCtxPopCurrent = hip.hipCtxPopCurrent
cuCtxPopCurrent_v2 = hip.hipCtxPopCurrent
cuCtxPushCurrent = hip.hipCtxPushCurrent
cuCtxPushCurrent_v2 = hip.hipCtxPushCurrent
cuCtxSetCurrent = hip.hipCtxSetCurrent
cuCtxGetCurrent = hip.hipCtxGetCurrent
cuCtxGetDevice = hip.hipCtxGetDevice
cuCtxGetApiVersion = hip.hipCtxGetApiVersion
cuCtxGetCacheConfig = hip.hipCtxGetCacheConfig
cuCtxSetCacheConfig = hip.hipCtxSetCacheConfig
cuCtxSetSharedMemConfig = hip.hipCtxSetSharedMemConfig
cuCtxGetSharedMemConfig = hip.hipCtxGetSharedMemConfig
cuCtxSynchronize = hip.hipCtxSynchronize
cuCtxGetFlags = hip.hipCtxGetFlags
cuCtxEnablePeerAccess = hip.hipCtxEnablePeerAccess
cuCtxDisablePeerAccess = hip.hipCtxDisablePeerAccess
cuDevicePrimaryCtxGetState = hip.hipDevicePrimaryCtxGetState
cuDevicePrimaryCtxRelease = hip.hipDevicePrimaryCtxRelease
cuDevicePrimaryCtxRelease_v2 = hip.hipDevicePrimaryCtxRelease
cuDevicePrimaryCtxRetain = hip.hipDevicePrimaryCtxRetain
cuDevicePrimaryCtxReset = hip.hipDevicePrimaryCtxReset
cuDevicePrimaryCtxReset_v2 = hip.hipDevicePrimaryCtxReset
cuDevicePrimaryCtxSetFlags = hip.hipDevicePrimaryCtxSetFlags
cuDevicePrimaryCtxSetFlags_v2 = hip.hipDevicePrimaryCtxSetFlags
cuModuleLoad = hip.hipModuleLoad
cuModuleUnload = hip.hipModuleUnload
cuModuleGetFunction = hip.hipModuleGetFunction
cudaFuncGetAttributes = hip.hipFuncGetAttributes
cuFuncGetAttribute = hip.hipFuncGetAttribute
cuModuleGetTexRef = hip.hipModuleGetTexRef
cuModuleLoadData = hip.hipModuleLoadData
cuModuleLoadDataEx = hip.hipModuleLoadDataEx
cuLaunchKernel = hip.hipModuleLaunchKernel
cudaLaunchCooperativeKernel = hip.hipLaunchCooperativeKernel
cudaLaunchCooperativeKernelMultiDevice = hip.hipLaunchCooperativeKernelMultiDevice
cuOccupancyMaxPotentialBlockSize = hip.hipModuleOccupancyMaxPotentialBlockSize
cuOccupancyMaxPotentialBlockSizeWithFlags = hip.hipModuleOccupancyMaxPotentialBlockSizeWithFlags
cuOccupancyMaxActiveBlocksPerMultiprocessor = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
cudaOccupancyMaxActiveBlocksPerMultiprocessor = hip.hipOccupancyMaxActiveBlocksPerMultiprocessor
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = hip.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
cudaOccupancyMaxPotentialBlockSize = hip.hipOccupancyMaxPotentialBlockSize
cuProfilerStart = hip.hipProfilerStart
cudaProfilerStart = hip.hipProfilerStart
cuProfilerStop = hip.hipProfilerStop
cudaProfilerStop = hip.hipProfilerStop
cudaConfigureCall = hip.hipConfigureCall
cudaSetupArgument = hip.hipSetupArgument
cudaLaunch = hip.hipLaunchByPtr
cudaLaunchKernel = hip.hipLaunchKernel
cuLaunchHostFunc = hip.hipLaunchHostFunc
cudaLaunchHostFunc = hip.hipLaunchHostFunc
cuMemcpy2DUnaligned = hip.hipDrvMemcpy2DUnaligned
cuMemcpy2DUnaligned_v2 = hip.hipDrvMemcpy2DUnaligned
cudaBindTextureToMipmappedArray = hip.hipBindTextureToMipmappedArray
cudaCreateTextureObject = hip.hipCreateTextureObject
cudaDestroyTextureObject = hip.hipDestroyTextureObject
cudaGetChannelDesc = hip.hipGetChannelDesc
cudaGetTextureObjectResourceDesc = hip.hipGetTextureObjectResourceDesc
cudaGetTextureObjectResourceViewDesc = hip.hipGetTextureObjectResourceViewDesc
cudaGetTextureObjectTextureDesc = hip.hipGetTextureObjectTextureDesc
cuTexObjectCreate = hip.hipTexObjectCreate
cuTexObjectDestroy = hip.hipTexObjectDestroy
cuTexObjectGetResourceDesc = hip.hipTexObjectGetResourceDesc
cuTexObjectGetResourceViewDesc = hip.hipTexObjectGetResourceViewDesc
cuTexObjectGetTextureDesc = hip.hipTexObjectGetTextureDesc
cudaGetTextureReference = hip.hipGetTextureReference
cuTexRefSetAddressMode = hip.hipTexRefSetAddressMode
cuTexRefSetArray = hip.hipTexRefSetArray
cuTexRefSetFilterMode = hip.hipTexRefSetFilterMode
cuTexRefSetFlags = hip.hipTexRefSetFlags
cuTexRefSetFormat = hip.hipTexRefSetFormat
cudaBindTexture = hip.hipBindTexture
cudaBindTexture2D = hip.hipBindTexture2D
cudaBindTextureToArray = hip.hipBindTextureToArray
cudaGetTextureAlignmentOffset = hip.hipGetTextureAlignmentOffset
cudaUnbindTexture = hip.hipUnbindTexture
cuTexRefGetAddress = hip.hipTexRefGetAddress
cuTexRefGetAddress_v2 = hip.hipTexRefGetAddress
cuTexRefGetAddressMode = hip.hipTexRefGetAddressMode
cuTexRefGetFilterMode = hip.hipTexRefGetFilterMode
cuTexRefGetFlags = hip.hipTexRefGetFlags
cuTexRefGetFormat = hip.hipTexRefGetFormat
cuTexRefGetMaxAnisotropy = hip.hipTexRefGetMaxAnisotropy
cuTexRefGetMipmapFilterMode = hip.hipTexRefGetMipmapFilterMode
cuTexRefGetMipmapLevelBias = hip.hipTexRefGetMipmapLevelBias
cuTexRefGetMipmapLevelClamp = hip.hipTexRefGetMipmapLevelClamp
cuTexRefGetMipmappedArray = hip.hipTexRefGetMipMappedArray
cuTexRefSetAddress = hip.hipTexRefSetAddress
cuTexRefSetAddress_v2 = hip.hipTexRefSetAddress
cuTexRefSetAddress2D = hip.hipTexRefSetAddress2D
cuTexRefSetAddress2D_v2 = hip.hipTexRefSetAddress2D
cuTexRefSetAddress2D_v3 = hip.hipTexRefSetAddress2D
cuTexRefSetMaxAnisotropy = hip.hipTexRefSetMaxAnisotropy
cuTexRefSetBorderColor = hip.hipTexRefSetBorderColor
cuTexRefSetMipmapFilterMode = hip.hipTexRefSetMipmapFilterMode
cuTexRefSetMipmapLevelBias = hip.hipTexRefSetMipmapLevelBias
cuTexRefSetMipmapLevelClamp = hip.hipTexRefSetMipmapLevelClamp
cuTexRefSetMipmappedArray = hip.hipTexRefSetMipmappedArray
cuMipmappedArrayCreate = hip.hipMipmappedArrayCreate
cuMipmappedArrayDestroy = hip.hipMipmappedArrayDestroy
cuMipmappedArrayGetLevel = hip.hipMipmappedArrayGetLevel
cuStreamBeginCapture = hip.hipStreamBeginCapture
cuStreamBeginCapture_v2 = hip.hipStreamBeginCapture
cudaStreamBeginCapture = hip.hipStreamBeginCapture
cuStreamEndCapture = hip.hipStreamEndCapture
cudaStreamEndCapture = hip.hipStreamEndCapture
cuStreamGetCaptureInfo = hip.hipStreamGetCaptureInfo
cudaStreamGetCaptureInfo = hip.hipStreamGetCaptureInfo
cuStreamGetCaptureInfo_v2 = hip.hipStreamGetCaptureInfo_v2
cuStreamIsCapturing = hip.hipStreamIsCapturing
cudaStreamIsCapturing = hip.hipStreamIsCapturing
cuStreamUpdateCaptureDependencies = hip.hipStreamUpdateCaptureDependencies
cuThreadExchangeStreamCaptureMode = hip.hipThreadExchangeStreamCaptureMode
cudaThreadExchangeStreamCaptureMode = hip.hipThreadExchangeStreamCaptureMode
cuGraphCreate = hip.hipGraphCreate
cudaGraphCreate = hip.hipGraphCreate
cuGraphDestroy = hip.hipGraphDestroy
cudaGraphDestroy = hip.hipGraphDestroy
cuGraphAddDependencies = hip.hipGraphAddDependencies
cudaGraphAddDependencies = hip.hipGraphAddDependencies
cuGraphRemoveDependencies = hip.hipGraphRemoveDependencies
cudaGraphRemoveDependencies = hip.hipGraphRemoveDependencies
cuGraphGetEdges = hip.hipGraphGetEdges
cudaGraphGetEdges = hip.hipGraphGetEdges
cuGraphGetNodes = hip.hipGraphGetNodes
cudaGraphGetNodes = hip.hipGraphGetNodes
cuGraphGetRootNodes = hip.hipGraphGetRootNodes
cudaGraphGetRootNodes = hip.hipGraphGetRootNodes
cuGraphNodeGetDependencies = hip.hipGraphNodeGetDependencies
cudaGraphNodeGetDependencies = hip.hipGraphNodeGetDependencies
cuGraphNodeGetDependentNodes = hip.hipGraphNodeGetDependentNodes
cudaGraphNodeGetDependentNodes = hip.hipGraphNodeGetDependentNodes
cuGraphNodeGetType = hip.hipGraphNodeGetType
cudaGraphNodeGetType = hip.hipGraphNodeGetType
cuGraphDestroyNode = hip.hipGraphDestroyNode
cudaGraphDestroyNode = hip.hipGraphDestroyNode
cuGraphClone = hip.hipGraphClone
cudaGraphClone = hip.hipGraphClone
cuGraphNodeFindInClone = hip.hipGraphNodeFindInClone
cudaGraphNodeFindInClone = hip.hipGraphNodeFindInClone
cuGraphInstantiate = hip.hipGraphInstantiate
cuGraphInstantiate_v2 = hip.hipGraphInstantiate
cudaGraphInstantiate = hip.hipGraphInstantiate
cuGraphInstantiateWithFlags = hip.hipGraphInstantiateWithFlags
cudaGraphInstantiateWithFlags = hip.hipGraphInstantiateWithFlags
cuGraphLaunch = hip.hipGraphLaunch
cudaGraphLaunch = hip.hipGraphLaunch
cuGraphUpload = hip.hipGraphUpload
cudaGraphUpload = hip.hipGraphUpload
cuGraphExecDestroy = hip.hipGraphExecDestroy
cudaGraphExecDestroy = hip.hipGraphExecDestroy
cuGraphExecUpdate = hip.hipGraphExecUpdate
cudaGraphExecUpdate = hip.hipGraphExecUpdate
cuGraphAddKernelNode = hip.hipGraphAddKernelNode
cudaGraphAddKernelNode = hip.hipGraphAddKernelNode
cuGraphKernelNodeGetParams = hip.hipGraphKernelNodeGetParams
cudaGraphKernelNodeGetParams = hip.hipGraphKernelNodeGetParams
cuGraphKernelNodeSetParams = hip.hipGraphKernelNodeSetParams
cudaGraphKernelNodeSetParams = hip.hipGraphKernelNodeSetParams
cuGraphExecKernelNodeSetParams = hip.hipGraphExecKernelNodeSetParams
cudaGraphExecKernelNodeSetParams = hip.hipGraphExecKernelNodeSetParams
cudaGraphAddMemcpyNode = hip.hipGraphAddMemcpyNode
cuGraphMemcpyNodeGetParams = hip.hipGraphMemcpyNodeGetParams
cudaGraphMemcpyNodeGetParams = hip.hipGraphMemcpyNodeGetParams
cuGraphMemcpyNodeSetParams = hip.hipGraphMemcpyNodeSetParams
cudaGraphMemcpyNodeSetParams = hip.hipGraphMemcpyNodeSetParams
cuGraphKernelNodeSetAttribute = hip.hipGraphKernelNodeSetAttribute
cudaGraphKernelNodeSetAttribute = hip.hipGraphKernelNodeSetAttribute
cuGraphKernelNodeGetAttribute = hip.hipGraphKernelNodeGetAttribute
cudaGraphKernelNodeGetAttribute = hip.hipGraphKernelNodeGetAttribute
cudaGraphExecMemcpyNodeSetParams = hip.hipGraphExecMemcpyNodeSetParams
cudaGraphAddMemcpyNode1D = hip.hipGraphAddMemcpyNode1D
cudaGraphMemcpyNodeSetParams1D = hip.hipGraphMemcpyNodeSetParams1D
cudaGraphExecMemcpyNodeSetParams1D = hip.hipGraphExecMemcpyNodeSetParams1D
cudaGraphAddMemcpyNodeFromSymbol = hip.hipGraphAddMemcpyNodeFromSymbol
cudaGraphMemcpyNodeSetParamsFromSymbol = hip.hipGraphMemcpyNodeSetParamsFromSymbol
cudaGraphExecMemcpyNodeSetParamsFromSymbol = hip.hipGraphExecMemcpyNodeSetParamsFromSymbol
cudaGraphAddMemcpyNodeToSymbol = hip.hipGraphAddMemcpyNodeToSymbol
cudaGraphMemcpyNodeSetParamsToSymbol = hip.hipGraphMemcpyNodeSetParamsToSymbol
cudaGraphExecMemcpyNodeSetParamsToSymbol = hip.hipGraphExecMemcpyNodeSetParamsToSymbol
cudaGraphAddMemsetNode = hip.hipGraphAddMemsetNode
cuGraphMemsetNodeGetParams = hip.hipGraphMemsetNodeGetParams
cudaGraphMemsetNodeGetParams = hip.hipGraphMemsetNodeGetParams
cuGraphMemsetNodeSetParams = hip.hipGraphMemsetNodeSetParams
cudaGraphMemsetNodeSetParams = hip.hipGraphMemsetNodeSetParams
cudaGraphExecMemsetNodeSetParams = hip.hipGraphExecMemsetNodeSetParams
cuGraphAddHostNode = hip.hipGraphAddHostNode
cudaGraphAddHostNode = hip.hipGraphAddHostNode
cuGraphHostNodeGetParams = hip.hipGraphHostNodeGetParams
cudaGraphHostNodeGetParams = hip.hipGraphHostNodeGetParams
cuGraphHostNodeSetParams = hip.hipGraphHostNodeSetParams
cudaGraphHostNodeSetParams = hip.hipGraphHostNodeSetParams
cuGraphExecHostNodeSetParams = hip.hipGraphExecHostNodeSetParams
cudaGraphExecHostNodeSetParams = hip.hipGraphExecHostNodeSetParams
cuGraphAddChildGraphNode = hip.hipGraphAddChildGraphNode
cudaGraphAddChildGraphNode = hip.hipGraphAddChildGraphNode
cuGraphChildGraphNodeGetGraph = hip.hipGraphChildGraphNodeGetGraph
cudaGraphChildGraphNodeGetGraph = hip.hipGraphChildGraphNodeGetGraph
cuGraphExecChildGraphNodeSetParams = hip.hipGraphExecChildGraphNodeSetParams
cudaGraphExecChildGraphNodeSetParams = hip.hipGraphExecChildGraphNodeSetParams
cuGraphAddEmptyNode = hip.hipGraphAddEmptyNode
cudaGraphAddEmptyNode = hip.hipGraphAddEmptyNode
cuGraphAddEventRecordNode = hip.hipGraphAddEventRecordNode
cudaGraphAddEventRecordNode = hip.hipGraphAddEventRecordNode
cuGraphEventRecordNodeGetEvent = hip.hipGraphEventRecordNodeGetEvent
cudaGraphEventRecordNodeGetEvent = hip.hipGraphEventRecordNodeGetEvent
cuGraphEventRecordNodeSetEvent = hip.hipGraphEventRecordNodeSetEvent
cudaGraphEventRecordNodeSetEvent = hip.hipGraphEventRecordNodeSetEvent
cuGraphExecEventRecordNodeSetEvent = hip.hipGraphExecEventRecordNodeSetEvent
cudaGraphExecEventRecordNodeSetEvent = hip.hipGraphExecEventRecordNodeSetEvent
cuGraphAddEventWaitNode = hip.hipGraphAddEventWaitNode
cudaGraphAddEventWaitNode = hip.hipGraphAddEventWaitNode
cuGraphEventWaitNodeGetEvent = hip.hipGraphEventWaitNodeGetEvent
cudaGraphEventWaitNodeGetEvent = hip.hipGraphEventWaitNodeGetEvent
cuGraphEventWaitNodeSetEvent = hip.hipGraphEventWaitNodeSetEvent
cudaGraphEventWaitNodeSetEvent = hip.hipGraphEventWaitNodeSetEvent
cuGraphExecEventWaitNodeSetEvent = hip.hipGraphExecEventWaitNodeSetEvent
cudaGraphExecEventWaitNodeSetEvent = hip.hipGraphExecEventWaitNodeSetEvent
cuDeviceGetGraphMemAttribute = hip.hipDeviceGetGraphMemAttribute
cudaDeviceGetGraphMemAttribute = hip.hipDeviceGetGraphMemAttribute
cuDeviceSetGraphMemAttribute = hip.hipDeviceSetGraphMemAttribute
cudaDeviceSetGraphMemAttribute = hip.hipDeviceSetGraphMemAttribute
cuDeviceGraphMemTrim = hip.hipDeviceGraphMemTrim
cudaDeviceGraphMemTrim = hip.hipDeviceGraphMemTrim
cuUserObjectCreate = hip.hipUserObjectCreate
cudaUserObjectCreate = hip.hipUserObjectCreate
cuUserObjectRelease = hip.hipUserObjectRelease
cudaUserObjectRelease = hip.hipUserObjectRelease
cuUserObjectRetain = hip.hipUserObjectRetain
cudaUserObjectRetain = hip.hipUserObjectRetain
cuGraphRetainUserObject = hip.hipGraphRetainUserObject
cudaGraphRetainUserObject = hip.hipGraphRetainUserObject
cuGraphReleaseUserObject = hip.hipGraphReleaseUserObject
cudaGraphReleaseUserObject = hip.hipGraphReleaseUserObject
cuMemAddressFree = hip.hipMemAddressFree
cuMemAddressReserve = hip.hipMemAddressReserve
cuMemCreate = hip.hipMemCreate
cuMemExportToShareableHandle = hip.hipMemExportToShareableHandle
cuMemGetAccess = hip.hipMemGetAccess
cuMemGetAllocationGranularity = hip.hipMemGetAllocationGranularity
cuMemGetAllocationPropertiesFromHandle = hip.hipMemGetAllocationPropertiesFromHandle
cuMemImportFromShareableHandle = hip.hipMemImportFromShareableHandle
cuMemMap = hip.hipMemMap
cuMemMapArrayAsync = hip.hipMemMapArrayAsync
cuMemRelease = hip.hipMemRelease
cuMemRetainAllocationHandle = hip.hipMemRetainAllocationHandle
cuMemSetAccess = hip.hipMemSetAccess
cuMemUnmap = hip.hipMemUnmap
cuGLGetDevices = hip.hipGLGetDevices
cudaGLGetDevices = hip.hipGLGetDevices
cuGraphicsGLRegisterBuffer = hip.hipGraphicsGLRegisterBuffer
cudaGraphicsGLRegisterBuffer = hip.hipGraphicsGLRegisterBuffer
cuGraphicsGLRegisterImage = hip.hipGraphicsGLRegisterImage
cudaGraphicsGLRegisterImage = hip.hipGraphicsGLRegisterImage
cuGraphicsMapResources = hip.hipGraphicsMapResources
cudaGraphicsMapResources = hip.hipGraphicsMapResources
cuGraphicsSubResourceGetMappedArray = hip.hipGraphicsSubResourceGetMappedArray
cudaGraphicsSubResourceGetMappedArray = hip.hipGraphicsSubResourceGetMappedArray
cuGraphicsResourceGetMappedPointer = hip.hipGraphicsResourceGetMappedPointer
cuGraphicsResourceGetMappedPointer_v2 = hip.hipGraphicsResourceGetMappedPointer
cudaGraphicsResourceGetMappedPointer = hip.hipGraphicsResourceGetMappedPointer
cuGraphicsUnmapResources = hip.hipGraphicsUnmapResources
cudaGraphicsUnmapResources = hip.hipGraphicsUnmapResources
cuGraphicsUnregisterResource = hip.hipGraphicsUnregisterResource
cudaGraphicsUnregisterResource = hip.hipGraphicsUnregisterResource