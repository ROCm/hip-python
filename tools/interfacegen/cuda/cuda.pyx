# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import os
import enum

import hip.hip

hip = hip.hip # makes hip types and routines accessible without import
                            # allows checks such as `hasattr(cuda.cuda,"hip")`

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

CU_TRSA_OVERRIDE_FORMAT = hip.hip.HIP_TRSA_OVERRIDE_FORMAT
CU_TRSF_READ_AS_INTEGER = hip.hip.HIP_TRSF_READ_AS_INTEGER
CU_TRSF_NORMALIZED_COORDINATES = hip.hip.HIP_TRSF_NORMALIZED_COORDINATES
CU_TRSF_SRGB = hip.hip.HIP_TRSF_SRGB
cudaTextureType1D = hip.hip.hipTextureType1D
cudaTextureType2D = hip.hip.hipTextureType2D
cudaTextureType3D = hip.hip.hipTextureType3D
cudaTextureTypeCubemap = hip.hip.hipTextureTypeCubemap
cudaTextureType1DLayered = hip.hip.hipTextureType1DLayered
cudaTextureType2DLayered = hip.hip.hipTextureType2DLayered
cudaTextureTypeCubemapLayered = hip.hip.hipTextureTypeCubemapLayered
CU_LAUNCH_PARAM_BUFFER_POINTER = hip.hip.HIP_LAUNCH_PARAM_BUFFER_POINTER
CU_LAUNCH_PARAM_BUFFER_SIZE = hip.hip.HIP_LAUNCH_PARAM_BUFFER_SIZE
CU_LAUNCH_PARAM_END = hip.hip.HIP_LAUNCH_PARAM_END
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = hip.hip.hipIpcMemLazyEnablePeerAccess
cudaIpcMemLazyEnablePeerAccess = hip.hip.hipIpcMemLazyEnablePeerAccess
CUDA_IPC_HANDLE_SIZE = hip.hip.HIP_IPC_HANDLE_SIZE
CU_IPC_HANDLE_SIZE = hip.hip.HIP_IPC_HANDLE_SIZE
CU_STREAM_DEFAULT = hip.hip.hipStreamDefault
cudaStreamDefault = hip.hip.hipStreamDefault
CU_STREAM_NON_BLOCKING = hip.hip.hipStreamNonBlocking
cudaStreamNonBlocking = hip.hip.hipStreamNonBlocking
CU_EVENT_DEFAULT = hip.hip.hipEventDefault
cudaEventDefault = hip.hip.hipEventDefault
CU_EVENT_BLOCKING_SYNC = hip.hip.hipEventBlockingSync
cudaEventBlockingSync = hip.hip.hipEventBlockingSync
CU_EVENT_DISABLE_TIMING = hip.hip.hipEventDisableTiming
cudaEventDisableTiming = hip.hip.hipEventDisableTiming
CU_EVENT_INTERPROCESS = hip.hip.hipEventInterprocess
cudaEventInterprocess = hip.hip.hipEventInterprocess
cudaHostAllocDefault = hip.hip.hipHostMallocDefault
CU_MEMHOSTALLOC_PORTABLE = hip.hip.hipHostMallocPortable
cudaHostAllocPortable = hip.hip.hipHostMallocPortable
CU_MEMHOSTALLOC_DEVICEMAP = hip.hip.hipHostMallocMapped
cudaHostAllocMapped = hip.hip.hipHostMallocMapped
CU_MEMHOSTALLOC_WRITECOMBINED = hip.hip.hipHostMallocWriteCombined
cudaHostAllocWriteCombined = hip.hip.hipHostMallocWriteCombined
CU_MEM_ATTACH_GLOBAL = hip.hip.hipMemAttachGlobal
cudaMemAttachGlobal = hip.hip.hipMemAttachGlobal
CU_MEM_ATTACH_HOST = hip.hip.hipMemAttachHost
cudaMemAttachHost = hip.hip.hipMemAttachHost
CU_MEM_ATTACH_SINGLE = hip.hip.hipMemAttachSingle
cudaMemAttachSingle = hip.hip.hipMemAttachSingle
cudaHostRegisterDefault = hip.hip.hipHostRegisterDefault
CU_MEMHOSTREGISTER_PORTABLE = hip.hip.hipHostRegisterPortable
cudaHostRegisterPortable = hip.hip.hipHostRegisterPortable
CU_MEMHOSTREGISTER_DEVICEMAP = hip.hip.hipHostRegisterMapped
cudaHostRegisterMapped = hip.hip.hipHostRegisterMapped
CU_MEMHOSTREGISTER_IOMEMORY = hip.hip.hipHostRegisterIoMemory
cudaHostRegisterIoMemory = hip.hip.hipHostRegisterIoMemory
CU_CTX_SCHED_AUTO = hip.hip.hipDeviceScheduleAuto
cudaDeviceScheduleAuto = hip.hip.hipDeviceScheduleAuto
CU_CTX_SCHED_SPIN = hip.hip.hipDeviceScheduleSpin
cudaDeviceScheduleSpin = hip.hip.hipDeviceScheduleSpin
CU_CTX_SCHED_YIELD = hip.hip.hipDeviceScheduleYield
cudaDeviceScheduleYield = hip.hip.hipDeviceScheduleYield
CU_CTX_BLOCKING_SYNC = hip.hip.hipDeviceScheduleBlockingSync
CU_CTX_SCHED_BLOCKING_SYNC = hip.hip.hipDeviceScheduleBlockingSync
cudaDeviceBlockingSync = hip.hip.hipDeviceScheduleBlockingSync
cudaDeviceScheduleBlockingSync = hip.hip.hipDeviceScheduleBlockingSync
CU_CTX_SCHED_MASK = hip.hip.hipDeviceScheduleMask
cudaDeviceScheduleMask = hip.hip.hipDeviceScheduleMask
CU_CTX_MAP_HOST = hip.hip.hipDeviceMapHost
cudaDeviceMapHost = hip.hip.hipDeviceMapHost
CU_CTX_LMEM_RESIZE_TO_MAX = hip.hip.hipDeviceLmemResizeToMax
cudaDeviceLmemResizeToMax = hip.hip.hipDeviceLmemResizeToMax
cudaArrayDefault = hip.hip.hipArrayDefault
CUDA_ARRAY3D_LAYERED = hip.hip.hipArrayLayered
cudaArrayLayered = hip.hip.hipArrayLayered
CUDA_ARRAY3D_SURFACE_LDST = hip.hip.hipArraySurfaceLoadStore
cudaArraySurfaceLoadStore = hip.hip.hipArraySurfaceLoadStore
CUDA_ARRAY3D_CUBEMAP = hip.hip.hipArrayCubemap
cudaArrayCubemap = hip.hip.hipArrayCubemap
CUDA_ARRAY3D_TEXTURE_GATHER = hip.hip.hipArrayTextureGather
cudaArrayTextureGather = hip.hip.hipArrayTextureGather
CU_OCCUPANCY_DEFAULT = hip.hip.hipOccupancyDefault
cudaOccupancyDefault = hip.hip.hipOccupancyDefault
CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC = hip.hip.hipCooperativeLaunchMultiDeviceNoPreSync
cudaCooperativeLaunchMultiDeviceNoPreSync = hip.hip.hipCooperativeLaunchMultiDeviceNoPreSync
CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC = hip.hip.hipCooperativeLaunchMultiDeviceNoPostSync
cudaCooperativeLaunchMultiDeviceNoPostSync = hip.hip.hipCooperativeLaunchMultiDeviceNoPostSync
CU_DEVICE_CPU = hip.hip.hipCpuDeviceId
cudaCpuDeviceId = hip.hip.hipCpuDeviceId
CU_DEVICE_INVALID = hip.hip.hipInvalidDeviceId
cudaInvalidDeviceId = hip.hip.hipInvalidDeviceId
CU_STREAM_WAIT_VALUE_GEQ = hip.hip.hipStreamWaitValueGte
CU_STREAM_WAIT_VALUE_EQ = hip.hip.hipStreamWaitValueEq
CU_STREAM_WAIT_VALUE_AND = hip.hip.hipStreamWaitValueAnd
CU_STREAM_WAIT_VALUE_NOR = hip.hip.hipStreamWaitValueNor
HIP_SUCCESS = hip.chip.HIP_SUCCESS
HIP_ERROR_INVALID_VALUE = hip.chip.HIP_ERROR_INVALID_VALUE
HIP_ERROR_NOT_INITIALIZED = hip.chip.HIP_ERROR_NOT_INITIALIZED
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = hip.chip.HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
cdef class CUuuid_st(hip.hip.hipUUID_t):
    pass
CUuuid = hip.hip.hipUUID
cudaUUID_t = hip.hip.hipUUID
cdef class cudaDeviceProp(hip.hip.hipDeviceProp_t):
    pass

HIP_PYTHON_CUmemorytype_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemorytype_HALLUCINATE_CONSTANTS","false")

class _CUmemorytype_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemorytype_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemorytype_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemorytype(enum.IntEnum,metaclass=_CUmemorytype_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemoryType
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

HIP_PYTHON_CUmemorytype_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemorytype_enum_HALLUCINATE_CONSTANTS","false")

class _CUmemorytype_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemorytype_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemorytype_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemorytype_enum(enum.IntEnum,metaclass=_CUmemorytype_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemoryType
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

HIP_PYTHON_cudaMemoryType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemoryType_HALLUCINATE_CONSTANTS","false")

class _cudaMemoryType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemoryType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemoryType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaMemoryType(enum.IntEnum,metaclass=_cudaMemoryType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemoryType
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

HIP_PYTHON_CUresult_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresult_HALLUCINATE_CONSTANTS","false")

class _CUresult_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresult_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresult_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUresult(enum.IntEnum,metaclass=_CUresult_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipError_t
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

HIP_PYTHON_cudaError_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaError_HALLUCINATE_CONSTANTS","false")

class _cudaError_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaError_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaError_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaError(enum.IntEnum,metaclass=_cudaError_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipError_t
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

HIP_PYTHON_cudaError_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaError_enum_HALLUCINATE_CONSTANTS","false")

class _cudaError_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaError_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaError_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaError_enum(enum.IntEnum,metaclass=_cudaError_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipError_t
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

HIP_PYTHON_cudaError_t_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaError_t_HALLUCINATE_CONSTANTS","false")

class _cudaError_t_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaError_t_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaError_t_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaError_t(enum.IntEnum,metaclass=_cudaError_t_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipError_t
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

HIP_PYTHON_CUdevice_attribute_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUdevice_attribute_HALLUCINATE_CONSTANTS","false")

class _CUdevice_attribute_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUdevice_attribute_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUdevice_attribute_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUdevice_attribute(enum.IntEnum,metaclass=_CUdevice_attribute_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipDeviceAttribute_t
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

HIP_PYTHON_CUdevice_attribute_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUdevice_attribute_enum_HALLUCINATE_CONSTANTS","false")

class _CUdevice_attribute_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUdevice_attribute_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUdevice_attribute_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUdevice_attribute_enum(enum.IntEnum,metaclass=_CUdevice_attribute_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipDeviceAttribute_t
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

HIP_PYTHON_cudaDeviceAttr_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaDeviceAttr_HALLUCINATE_CONSTANTS","false")

class _cudaDeviceAttr_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaDeviceAttr_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaDeviceAttr_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaDeviceAttr(enum.IntEnum,metaclass=_cudaDeviceAttr_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipDeviceAttribute_t
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

HIP_PYTHON_CUcomputemode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUcomputemode_HALLUCINATE_CONSTANTS","false")

class _CUcomputemode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUcomputemode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUcomputemode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUcomputemode(enum.IntEnum,metaclass=_CUcomputemode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipComputeMode
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

HIP_PYTHON_CUcomputemode_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUcomputemode_enum_HALLUCINATE_CONSTANTS","false")

class _CUcomputemode_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUcomputemode_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUcomputemode_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUcomputemode_enum(enum.IntEnum,metaclass=_CUcomputemode_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipComputeMode
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

HIP_PYTHON_cudaComputeMode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaComputeMode_HALLUCINATE_CONSTANTS","false")

class _cudaComputeMode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaComputeMode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaComputeMode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaComputeMode(enum.IntEnum,metaclass=_cudaComputeMode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipComputeMode
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

HIP_PYTHON_cudaChannelFormatKind_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaChannelFormatKind_HALLUCINATE_CONSTANTS","false")

class _cudaChannelFormatKind_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaChannelFormatKind_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaChannelFormatKind_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaChannelFormatKind(enum.IntEnum,metaclass=_cudaChannelFormatKind_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipChannelFormatKind
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

HIP_PYTHON_CUarray_format_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUarray_format_HALLUCINATE_CONSTANTS","false")

class _CUarray_format_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUarray_format_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUarray_format_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUarray_format(enum.IntEnum,metaclass=_CUarray_format_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipArray_Format
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

HIP_PYTHON_CUarray_format_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUarray_format_enum_HALLUCINATE_CONSTANTS","false")

class _CUarray_format_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUarray_format_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUarray_format_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUarray_format_enum(enum.IntEnum,metaclass=_CUarray_format_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipArray_Format
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
CUarray = hip.hip.hipArray_t
cudaArray_t = hip.hip.hipArray_t
cudaArray_const_t = hip.hip.hipArray_const_t
cdef class CUmipmappedArray_st(hip.hip.hipMipmappedArray):
    pass
cdef class cudaMipmappedArray(hip.hip.hipMipmappedArray):
    pass
CUmipmappedArray = hip.hip.hipMipmappedArray_t
cudaMipmappedArray_t = hip.hip.hipMipmappedArray_t
cudaMipmappedArray_const_t = hip.hip.hipMipmappedArray_const_t

HIP_PYTHON_cudaResourceType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaResourceType_HALLUCINATE_CONSTANTS","false")

class _cudaResourceType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaResourceType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaResourceType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaResourceType(enum.IntEnum,metaclass=_cudaResourceType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipResourceType
    cudaResourceTypeArray = hip.chip.hipResourceTypeArray
    hipResourceTypeArray = hip.chip.hipResourceTypeArray
    cudaResourceTypeMipmappedArray = hip.chip.hipResourceTypeMipmappedArray
    hipResourceTypeMipmappedArray = hip.chip.hipResourceTypeMipmappedArray
    cudaResourceTypeLinear = hip.chip.hipResourceTypeLinear
    hipResourceTypeLinear = hip.chip.hipResourceTypeLinear
    cudaResourceTypePitch2D = hip.chip.hipResourceTypePitch2D
    hipResourceTypePitch2D = hip.chip.hipResourceTypePitch2D

HIP_PYTHON_CUresourcetype_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresourcetype_enum_HALLUCINATE_CONSTANTS","false")

class _CUresourcetype_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresourcetype_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresourcetype_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUresourcetype_enum(enum.IntEnum,metaclass=_CUresourcetype_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.HIPresourcetype_enum
    CU_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    HIP_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    CU_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    HIP_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    CU_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D
    HIP_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D

HIP_PYTHON_CUresourcetype_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresourcetype_HALLUCINATE_CONSTANTS","false")

class _CUresourcetype_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresourcetype_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresourcetype_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUresourcetype(enum.IntEnum,metaclass=_CUresourcetype_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.HIPresourcetype
    CU_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    HIP_RESOURCE_TYPE_ARRAY = hip.chip.HIP_RESOURCE_TYPE_ARRAY
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = hip.chip.HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
    CU_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    HIP_RESOURCE_TYPE_LINEAR = hip.chip.HIP_RESOURCE_TYPE_LINEAR
    CU_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D
    HIP_RESOURCE_TYPE_PITCH2D = hip.chip.HIP_RESOURCE_TYPE_PITCH2D

HIP_PYTHON_CUaddress_mode_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUaddress_mode_enum_HALLUCINATE_CONSTANTS","false")

class _CUaddress_mode_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUaddress_mode_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUaddress_mode_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUaddress_mode_enum(enum.IntEnum,metaclass=_CUaddress_mode_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.HIPaddress_mode_enum
    CU_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    HIP_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    CU_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    HIP_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    CU_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    HIP_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    CU_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER
    HIP_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER

HIP_PYTHON_CUaddress_mode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUaddress_mode_HALLUCINATE_CONSTANTS","false")

class _CUaddress_mode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUaddress_mode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUaddress_mode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUaddress_mode(enum.IntEnum,metaclass=_CUaddress_mode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.HIPaddress_mode
    CU_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    HIP_TR_ADDRESS_MODE_WRAP = hip.chip.HIP_TR_ADDRESS_MODE_WRAP
    CU_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    HIP_TR_ADDRESS_MODE_CLAMP = hip.chip.HIP_TR_ADDRESS_MODE_CLAMP
    CU_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    HIP_TR_ADDRESS_MODE_MIRROR = hip.chip.HIP_TR_ADDRESS_MODE_MIRROR
    CU_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER
    HIP_TR_ADDRESS_MODE_BORDER = hip.chip.HIP_TR_ADDRESS_MODE_BORDER

HIP_PYTHON_CUfilter_mode_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfilter_mode_enum_HALLUCINATE_CONSTANTS","false")

class _CUfilter_mode_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfilter_mode_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfilter_mode_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUfilter_mode_enum(enum.IntEnum,metaclass=_CUfilter_mode_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.HIPfilter_mode_enum
    CU_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    HIP_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    CU_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR
    HIP_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR

HIP_PYTHON_CUfilter_mode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfilter_mode_HALLUCINATE_CONSTANTS","false")

class _CUfilter_mode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfilter_mode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfilter_mode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUfilter_mode(enum.IntEnum,metaclass=_CUfilter_mode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.HIPfilter_mode
    CU_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    HIP_TR_FILTER_MODE_POINT = hip.chip.HIP_TR_FILTER_MODE_POINT
    CU_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR
    HIP_TR_FILTER_MODE_LINEAR = hip.chip.HIP_TR_FILTER_MODE_LINEAR
cdef class CUDA_TEXTURE_DESC_st(hip.hip.HIP_TEXTURE_DESC_st):
    pass
CUDA_TEXTURE_DESC = hip.hip.HIP_TEXTURE_DESC
CUDA_TEXTURE_DESC_v1 = hip.hip.HIP_TEXTURE_DESC

HIP_PYTHON_cudaResourceViewFormat_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaResourceViewFormat_HALLUCINATE_CONSTANTS","false")

class _cudaResourceViewFormat_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaResourceViewFormat_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaResourceViewFormat_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaResourceViewFormat(enum.IntEnum,metaclass=_cudaResourceViewFormat_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipResourceViewFormat
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

HIP_PYTHON_CUresourceViewFormat_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresourceViewFormat_enum_HALLUCINATE_CONSTANTS","false")

class _CUresourceViewFormat_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresourceViewFormat_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresourceViewFormat_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUresourceViewFormat_enum(enum.IntEnum,metaclass=_CUresourceViewFormat_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.HIPresourceViewFormat_enum
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

HIP_PYTHON_CUresourceViewFormat_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUresourceViewFormat_HALLUCINATE_CONSTANTS","false")

class _CUresourceViewFormat_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUresourceViewFormat_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUresourceViewFormat_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUresourceViewFormat(enum.IntEnum,metaclass=_CUresourceViewFormat_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.HIPresourceViewFormat
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
CUDA_RESOURCE_DESC = hip.hip.HIP_RESOURCE_DESC
CUDA_RESOURCE_DESC_v1 = hip.hip.HIP_RESOURCE_DESC
cdef class cudaResourceViewDesc(hip.hip.hipResourceViewDesc):
    pass
cdef class CUDA_RESOURCE_VIEW_DESC_st(hip.hip.HIP_RESOURCE_VIEW_DESC_st):
    pass
CUDA_RESOURCE_VIEW_DESC = hip.hip.HIP_RESOURCE_VIEW_DESC
CUDA_RESOURCE_VIEW_DESC_v1 = hip.hip.HIP_RESOURCE_VIEW_DESC

HIP_PYTHON_cudaMemcpyKind_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemcpyKind_HALLUCINATE_CONSTANTS","false")

class _cudaMemcpyKind_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemcpyKind_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemcpyKind_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaMemcpyKind(enum.IntEnum,metaclass=_cudaMemcpyKind_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemcpyKind
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

HIP_PYTHON_CUfunction_attribute_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfunction_attribute_HALLUCINATE_CONSTANTS","false")

class _CUfunction_attribute_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfunction_attribute_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfunction_attribute_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUfunction_attribute(enum.IntEnum,metaclass=_CUfunction_attribute_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipFunction_attribute
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

HIP_PYTHON_CUfunction_attribute_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfunction_attribute_enum_HALLUCINATE_CONSTANTS","false")

class _CUfunction_attribute_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfunction_attribute_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfunction_attribute_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUfunction_attribute_enum(enum.IntEnum,metaclass=_CUfunction_attribute_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipFunction_attribute
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

HIP_PYTHON_CUpointer_attribute_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUpointer_attribute_HALLUCINATE_CONSTANTS","false")

class _CUpointer_attribute_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUpointer_attribute_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUpointer_attribute_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUpointer_attribute(enum.IntEnum,metaclass=_CUpointer_attribute_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipPointer_attribute
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

HIP_PYTHON_CUpointer_attribute_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUpointer_attribute_enum_HALLUCINATE_CONSTANTS","false")

class _CUpointer_attribute_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUpointer_attribute_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUpointer_attribute_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUpointer_attribute_enum(enum.IntEnum,metaclass=_CUpointer_attribute_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipPointer_attribute
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
cudaCreateChannelDesc = hip.hip.hipCreateChannelDesc
CUtexObject = hip.hip.hipTextureObject_t
CUtexObject_v1 = hip.hip.hipTextureObject_t
cudaTextureObject_t = hip.hip.hipTextureObject_t

HIP_PYTHON_cudaTextureAddressMode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaTextureAddressMode_HALLUCINATE_CONSTANTS","false")

class _cudaTextureAddressMode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaTextureAddressMode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaTextureAddressMode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaTextureAddressMode(enum.IntEnum,metaclass=_cudaTextureAddressMode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipTextureAddressMode
    cudaAddressModeWrap = hip.chip.hipAddressModeWrap
    hipAddressModeWrap = hip.chip.hipAddressModeWrap
    cudaAddressModeClamp = hip.chip.hipAddressModeClamp
    hipAddressModeClamp = hip.chip.hipAddressModeClamp
    cudaAddressModeMirror = hip.chip.hipAddressModeMirror
    hipAddressModeMirror = hip.chip.hipAddressModeMirror
    cudaAddressModeBorder = hip.chip.hipAddressModeBorder
    hipAddressModeBorder = hip.chip.hipAddressModeBorder

HIP_PYTHON_cudaTextureFilterMode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaTextureFilterMode_HALLUCINATE_CONSTANTS","false")

class _cudaTextureFilterMode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaTextureFilterMode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaTextureFilterMode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaTextureFilterMode(enum.IntEnum,metaclass=_cudaTextureFilterMode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipTextureFilterMode
    cudaFilterModePoint = hip.chip.hipFilterModePoint
    hipFilterModePoint = hip.chip.hipFilterModePoint
    cudaFilterModeLinear = hip.chip.hipFilterModeLinear
    hipFilterModeLinear = hip.chip.hipFilterModeLinear

HIP_PYTHON_cudaTextureReadMode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaTextureReadMode_HALLUCINATE_CONSTANTS","false")

class _cudaTextureReadMode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaTextureReadMode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaTextureReadMode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaTextureReadMode(enum.IntEnum,metaclass=_cudaTextureReadMode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipTextureReadMode
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
CUsurfObject = hip.hip.hipSurfaceObject_t
CUsurfObject_v1 = hip.hip.hipSurfaceObject_t
cudaSurfaceObject_t = hip.hip.hipSurfaceObject_t
cdef class surfaceReference(hip.hip.surfaceReference):
    pass

HIP_PYTHON_cudaSurfaceBoundaryMode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaSurfaceBoundaryMode_HALLUCINATE_CONSTANTS","false")

class _cudaSurfaceBoundaryMode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaSurfaceBoundaryMode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaSurfaceBoundaryMode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaSurfaceBoundaryMode(enum.IntEnum,metaclass=_cudaSurfaceBoundaryMode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipSurfaceBoundaryMode
    cudaBoundaryModeZero = hip.chip.hipBoundaryModeZero
    hipBoundaryModeZero = hip.chip.hipBoundaryModeZero
    cudaBoundaryModeTrap = hip.chip.hipBoundaryModeTrap
    hipBoundaryModeTrap = hip.chip.hipBoundaryModeTrap
    cudaBoundaryModeClamp = hip.chip.hipBoundaryModeClamp
    hipBoundaryModeClamp = hip.chip.hipBoundaryModeClamp
cdef class CUctx_st(hip.hip.ihipCtx_t):
    pass
CUcontext = hip.hip.hipCtx_t

HIP_PYTHON_CUdevice_P2PAttribute_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUdevice_P2PAttribute_HALLUCINATE_CONSTANTS","false")

class _CUdevice_P2PAttribute_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUdevice_P2PAttribute_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUdevice_P2PAttribute_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUdevice_P2PAttribute(enum.IntEnum,metaclass=_CUdevice_P2PAttribute_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipDeviceP2PAttr
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

HIP_PYTHON_CUdevice_P2PAttribute_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUdevice_P2PAttribute_enum_HALLUCINATE_CONSTANTS","false")

class _CUdevice_P2PAttribute_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUdevice_P2PAttribute_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUdevice_P2PAttribute_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUdevice_P2PAttribute_enum(enum.IntEnum,metaclass=_CUdevice_P2PAttribute_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipDeviceP2PAttr
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

HIP_PYTHON_cudaDeviceP2PAttr_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaDeviceP2PAttr_HALLUCINATE_CONSTANTS","false")

class _cudaDeviceP2PAttr_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaDeviceP2PAttr_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaDeviceP2PAttr_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaDeviceP2PAttr(enum.IntEnum,metaclass=_cudaDeviceP2PAttr_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipDeviceP2PAttr
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
CUstream = hip.hip.hipStream_t
cudaStream_t = hip.hip.hipStream_t
cdef class CUipcMemHandle_st(hip.hip.hipIpcMemHandle_st):
    pass
cdef class cudaIpcMemHandle_st(hip.hip.hipIpcMemHandle_st):
    pass
CUipcMemHandle = hip.hip.hipIpcMemHandle_t
CUipcMemHandle_v1 = hip.hip.hipIpcMemHandle_t
cudaIpcMemHandle_t = hip.hip.hipIpcMemHandle_t
cdef class CUipcEventHandle_st(hip.hip.hipIpcEventHandle_st):
    pass
cdef class cudaIpcEventHandle_st(hip.hip.hipIpcEventHandle_st):
    pass
CUipcEventHandle = hip.hip.hipIpcEventHandle_t
CUipcEventHandle_v1 = hip.hip.hipIpcEventHandle_t
cudaIpcEventHandle_t = hip.hip.hipIpcEventHandle_t
cdef class CUmod_st(hip.hip.ihipModule_t):
    pass
CUmodule = hip.hip.hipModule_t
cdef class CUfunc_st(hip.hip.ihipModuleSymbol_t):
    pass
CUfunction = hip.hip.hipFunction_t
cudaFunction_t = hip.hip.hipFunction_t
cdef class CUmemPoolHandle_st(hip.hip.ihipMemPoolHandle_t):
    pass
CUmemoryPool = hip.hip.hipMemPool_t
cudaMemPool_t = hip.hip.hipMemPool_t
cdef class cudaFuncAttributes(hip.hip.hipFuncAttributes):
    pass
cdef class CUevent_st(hip.hip.ihipEvent_t):
    pass
CUevent = hip.hip.hipEvent_t
cudaEvent_t = hip.hip.hipEvent_t

HIP_PYTHON_CUlimit_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUlimit_HALLUCINATE_CONSTANTS","false")

class _CUlimit_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUlimit_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUlimit_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUlimit(enum.IntEnum,metaclass=_CUlimit_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipLimit_t
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

HIP_PYTHON_CUlimit_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUlimit_enum_HALLUCINATE_CONSTANTS","false")

class _CUlimit_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUlimit_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUlimit_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUlimit_enum(enum.IntEnum,metaclass=_CUlimit_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipLimit_t
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

HIP_PYTHON_cudaLimit_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaLimit_HALLUCINATE_CONSTANTS","false")

class _cudaLimit_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaLimit_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaLimit_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaLimit(enum.IntEnum,metaclass=_cudaLimit_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipLimit_t
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

HIP_PYTHON_CUmem_advise_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmem_advise_HALLUCINATE_CONSTANTS","false")

class _CUmem_advise_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmem_advise_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmem_advise_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmem_advise(enum.IntEnum,metaclass=_CUmem_advise_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemoryAdvise
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

HIP_PYTHON_CUmem_advise_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmem_advise_enum_HALLUCINATE_CONSTANTS","false")

class _CUmem_advise_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmem_advise_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmem_advise_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmem_advise_enum(enum.IntEnum,metaclass=_CUmem_advise_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemoryAdvise
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

HIP_PYTHON_cudaMemoryAdvise_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemoryAdvise_HALLUCINATE_CONSTANTS","false")

class _cudaMemoryAdvise_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemoryAdvise_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemoryAdvise_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaMemoryAdvise(enum.IntEnum,metaclass=_cudaMemoryAdvise_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemoryAdvise
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

HIP_PYTHON_CUmem_range_attribute_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmem_range_attribute_HALLUCINATE_CONSTANTS","false")

class _CUmem_range_attribute_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmem_range_attribute_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmem_range_attribute_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmem_range_attribute(enum.IntEnum,metaclass=_CUmem_range_attribute_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemRangeAttribute
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

HIP_PYTHON_CUmem_range_attribute_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmem_range_attribute_enum_HALLUCINATE_CONSTANTS","false")

class _CUmem_range_attribute_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmem_range_attribute_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmem_range_attribute_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmem_range_attribute_enum(enum.IntEnum,metaclass=_CUmem_range_attribute_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemRangeAttribute
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

HIP_PYTHON_cudaMemRangeAttribute_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemRangeAttribute_HALLUCINATE_CONSTANTS","false")

class _cudaMemRangeAttribute_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemRangeAttribute_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemRangeAttribute_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaMemRangeAttribute(enum.IntEnum,metaclass=_cudaMemRangeAttribute_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemRangeAttribute
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

HIP_PYTHON_CUmemPool_attribute_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemPool_attribute_HALLUCINATE_CONSTANTS","false")

class _CUmemPool_attribute_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemPool_attribute_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemPool_attribute_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemPool_attribute(enum.IntEnum,metaclass=_CUmemPool_attribute_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemPoolAttr
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

HIP_PYTHON_CUmemPool_attribute_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemPool_attribute_enum_HALLUCINATE_CONSTANTS","false")

class _CUmemPool_attribute_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemPool_attribute_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemPool_attribute_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemPool_attribute_enum(enum.IntEnum,metaclass=_CUmemPool_attribute_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemPoolAttr
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

HIP_PYTHON_cudaMemPoolAttr_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemPoolAttr_HALLUCINATE_CONSTANTS","false")

class _cudaMemPoolAttr_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemPoolAttr_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemPoolAttr_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaMemPoolAttr(enum.IntEnum,metaclass=_cudaMemPoolAttr_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemPoolAttr
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

HIP_PYTHON_CUmemLocationType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemLocationType_HALLUCINATE_CONSTANTS","false")

class _CUmemLocationType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemLocationType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemLocationType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemLocationType(enum.IntEnum,metaclass=_CUmemLocationType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemLocationType
    CU_MEM_LOCATION_TYPE_INVALID = hip.chip.hipMemLocationTypeInvalid
    cudaMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    hipMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    CU_MEM_LOCATION_TYPE_DEVICE = hip.chip.hipMemLocationTypeDevice
    cudaMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice
    hipMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice

HIP_PYTHON_CUmemLocationType_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemLocationType_enum_HALLUCINATE_CONSTANTS","false")

class _CUmemLocationType_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemLocationType_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemLocationType_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemLocationType_enum(enum.IntEnum,metaclass=_CUmemLocationType_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemLocationType
    CU_MEM_LOCATION_TYPE_INVALID = hip.chip.hipMemLocationTypeInvalid
    cudaMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    hipMemLocationTypeInvalid = hip.chip.hipMemLocationTypeInvalid
    CU_MEM_LOCATION_TYPE_DEVICE = hip.chip.hipMemLocationTypeDevice
    cudaMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice
    hipMemLocationTypeDevice = hip.chip.hipMemLocationTypeDevice

HIP_PYTHON_cudaMemLocationType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemLocationType_HALLUCINATE_CONSTANTS","false")

class _cudaMemLocationType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemLocationType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemLocationType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaMemLocationType(enum.IntEnum,metaclass=_cudaMemLocationType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemLocationType
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

HIP_PYTHON_CUmemAccess_flags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAccess_flags_HALLUCINATE_CONSTANTS","false")

class _CUmemAccess_flags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAccess_flags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAccess_flags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemAccess_flags(enum.IntEnum,metaclass=_CUmemAccess_flags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAccessFlags
    CU_MEM_ACCESS_FLAGS_PROT_NONE = hip.chip.hipMemAccessFlagsProtNone
    cudaMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    hipMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    CU_MEM_ACCESS_FLAGS_PROT_READ = hip.chip.hipMemAccessFlagsProtRead
    cudaMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    hipMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hip.chip.hipMemAccessFlagsProtReadWrite
    cudaMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite
    hipMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite

HIP_PYTHON_CUmemAccess_flags_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAccess_flags_enum_HALLUCINATE_CONSTANTS","false")

class _CUmemAccess_flags_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAccess_flags_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAccess_flags_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemAccess_flags_enum(enum.IntEnum,metaclass=_CUmemAccess_flags_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAccessFlags
    CU_MEM_ACCESS_FLAGS_PROT_NONE = hip.chip.hipMemAccessFlagsProtNone
    cudaMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    hipMemAccessFlagsProtNone = hip.chip.hipMemAccessFlagsProtNone
    CU_MEM_ACCESS_FLAGS_PROT_READ = hip.chip.hipMemAccessFlagsProtRead
    cudaMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    hipMemAccessFlagsProtRead = hip.chip.hipMemAccessFlagsProtRead
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hip.chip.hipMemAccessFlagsProtReadWrite
    cudaMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite
    hipMemAccessFlagsProtReadWrite = hip.chip.hipMemAccessFlagsProtReadWrite

HIP_PYTHON_cudaMemAccessFlags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemAccessFlags_HALLUCINATE_CONSTANTS","false")

class _cudaMemAccessFlags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemAccessFlags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemAccessFlags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaMemAccessFlags(enum.IntEnum,metaclass=_cudaMemAccessFlags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAccessFlags
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

HIP_PYTHON_CUmemAllocationType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationType_HALLUCINATE_CONSTANTS","false")

class _CUmemAllocationType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemAllocationType(enum.IntEnum,metaclass=_CUmemAllocationType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAllocationType
    CU_MEM_ALLOCATION_TYPE_INVALID = hip.chip.hipMemAllocationTypeInvalid
    cudaMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    hipMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    CU_MEM_ALLOCATION_TYPE_PINNED = hip.chip.hipMemAllocationTypePinned
    cudaMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    hipMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    CU_MEM_ALLOCATION_TYPE_MAX = hip.chip.hipMemAllocationTypeMax
    cudaMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax
    hipMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax

HIP_PYTHON_CUmemAllocationType_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationType_enum_HALLUCINATE_CONSTANTS","false")

class _CUmemAllocationType_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationType_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationType_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemAllocationType_enum(enum.IntEnum,metaclass=_CUmemAllocationType_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAllocationType
    CU_MEM_ALLOCATION_TYPE_INVALID = hip.chip.hipMemAllocationTypeInvalid
    cudaMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    hipMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    CU_MEM_ALLOCATION_TYPE_PINNED = hip.chip.hipMemAllocationTypePinned
    cudaMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    hipMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    CU_MEM_ALLOCATION_TYPE_MAX = hip.chip.hipMemAllocationTypeMax
    cudaMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax
    hipMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax

HIP_PYTHON_cudaMemAllocationType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemAllocationType_HALLUCINATE_CONSTANTS","false")

class _cudaMemAllocationType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemAllocationType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemAllocationType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaMemAllocationType(enum.IntEnum,metaclass=_cudaMemAllocationType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAllocationType
    CU_MEM_ALLOCATION_TYPE_INVALID = hip.chip.hipMemAllocationTypeInvalid
    cudaMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    hipMemAllocationTypeInvalid = hip.chip.hipMemAllocationTypeInvalid
    CU_MEM_ALLOCATION_TYPE_PINNED = hip.chip.hipMemAllocationTypePinned
    cudaMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    hipMemAllocationTypePinned = hip.chip.hipMemAllocationTypePinned
    CU_MEM_ALLOCATION_TYPE_MAX = hip.chip.hipMemAllocationTypeMax
    cudaMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax
    hipMemAllocationTypeMax = hip.chip.hipMemAllocationTypeMax

HIP_PYTHON_CUmemAllocationHandleType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationHandleType_HALLUCINATE_CONSTANTS","false")

class _CUmemAllocationHandleType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationHandleType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationHandleType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemAllocationHandleType(enum.IntEnum,metaclass=_CUmemAllocationHandleType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAllocationHandleType
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

HIP_PYTHON_CUmemAllocationHandleType_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationHandleType_enum_HALLUCINATE_CONSTANTS","false")

class _CUmemAllocationHandleType_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationHandleType_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationHandleType_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemAllocationHandleType_enum(enum.IntEnum,metaclass=_CUmemAllocationHandleType_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAllocationHandleType
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

HIP_PYTHON_cudaMemAllocationHandleType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaMemAllocationHandleType_HALLUCINATE_CONSTANTS","false")

class _cudaMemAllocationHandleType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaMemAllocationHandleType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaMemAllocationHandleType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaMemAllocationHandleType(enum.IntEnum,metaclass=_cudaMemAllocationHandleType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAllocationHandleType
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

HIP_PYTHON_CUjit_option_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUjit_option_HALLUCINATE_CONSTANTS","false")

class _CUjit_option_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUjit_option_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUjit_option_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUjit_option(enum.IntEnum,metaclass=_CUjit_option_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipJitOption
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

HIP_PYTHON_CUjit_option_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUjit_option_enum_HALLUCINATE_CONSTANTS","false")

class _CUjit_option_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUjit_option_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUjit_option_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUjit_option_enum(enum.IntEnum,metaclass=_CUjit_option_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipJitOption
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

HIP_PYTHON_cudaFuncAttribute_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaFuncAttribute_HALLUCINATE_CONSTANTS","false")

class _cudaFuncAttribute_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaFuncAttribute_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaFuncAttribute_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaFuncAttribute(enum.IntEnum,metaclass=_cudaFuncAttribute_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipFuncAttribute
    cudaFuncAttributeMaxDynamicSharedMemorySize = hip.chip.hipFuncAttributeMaxDynamicSharedMemorySize
    hipFuncAttributeMaxDynamicSharedMemorySize = hip.chip.hipFuncAttributeMaxDynamicSharedMemorySize
    cudaFuncAttributePreferredSharedMemoryCarveout = hip.chip.hipFuncAttributePreferredSharedMemoryCarveout
    hipFuncAttributePreferredSharedMemoryCarveout = hip.chip.hipFuncAttributePreferredSharedMemoryCarveout
    cudaFuncAttributeMax = hip.chip.hipFuncAttributeMax
    hipFuncAttributeMax = hip.chip.hipFuncAttributeMax

HIP_PYTHON_CUfunc_cache_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfunc_cache_HALLUCINATE_CONSTANTS","false")

class _CUfunc_cache_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfunc_cache_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfunc_cache_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUfunc_cache(enum.IntEnum,metaclass=_CUfunc_cache_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipFuncCache_t
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

HIP_PYTHON_CUfunc_cache_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUfunc_cache_enum_HALLUCINATE_CONSTANTS","false")

class _CUfunc_cache_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUfunc_cache_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUfunc_cache_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUfunc_cache_enum(enum.IntEnum,metaclass=_CUfunc_cache_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipFuncCache_t
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

HIP_PYTHON_cudaFuncCache_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaFuncCache_HALLUCINATE_CONSTANTS","false")

class _cudaFuncCache_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaFuncCache_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaFuncCache_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaFuncCache(enum.IntEnum,metaclass=_cudaFuncCache_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipFuncCache_t
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

HIP_PYTHON_CUsharedconfig_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUsharedconfig_HALLUCINATE_CONSTANTS","false")

class _CUsharedconfig_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUsharedconfig_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUsharedconfig_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUsharedconfig(enum.IntEnum,metaclass=_CUsharedconfig_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipSharedMemConfig
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = hip.chip.hipSharedMemBankSizeDefault
    cudaSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    hipSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeFourByte
    cudaSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    hipSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeEightByte
    cudaSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
    hipSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte

HIP_PYTHON_CUsharedconfig_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUsharedconfig_enum_HALLUCINATE_CONSTANTS","false")

class _CUsharedconfig_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUsharedconfig_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUsharedconfig_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUsharedconfig_enum(enum.IntEnum,metaclass=_CUsharedconfig_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipSharedMemConfig
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = hip.chip.hipSharedMemBankSizeDefault
    cudaSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    hipSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeFourByte
    cudaSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    hipSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeEightByte
    cudaSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
    hipSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte

HIP_PYTHON_cudaSharedMemConfig_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaSharedMemConfig_HALLUCINATE_CONSTANTS","false")

class _cudaSharedMemConfig_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaSharedMemConfig_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaSharedMemConfig_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaSharedMemConfig(enum.IntEnum,metaclass=_cudaSharedMemConfig_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipSharedMemConfig
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = hip.chip.hipSharedMemBankSizeDefault
    cudaSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    hipSharedMemBankSizeDefault = hip.chip.hipSharedMemBankSizeDefault
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeFourByte
    cudaSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    hipSharedMemBankSizeFourByte = hip.chip.hipSharedMemBankSizeFourByte
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = hip.chip.hipSharedMemBankSizeEightByte
    cudaSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
    hipSharedMemBankSizeEightByte = hip.chip.hipSharedMemBankSizeEightByte
cudaLaunchParams = hip.hip.hipLaunchParams

HIP_PYTHON_CUexternalMemoryHandleType_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUexternalMemoryHandleType_enum_HALLUCINATE_CONSTANTS","false")

class _CUexternalMemoryHandleType_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUexternalMemoryHandleType_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUexternalMemoryHandleType_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUexternalMemoryHandleType_enum(enum.IntEnum,metaclass=_CUexternalMemoryHandleType_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipExternalMemoryHandleType_enum
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

HIP_PYTHON_CUexternalMemoryHandleType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUexternalMemoryHandleType_HALLUCINATE_CONSTANTS","false")

class _CUexternalMemoryHandleType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUexternalMemoryHandleType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUexternalMemoryHandleType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUexternalMemoryHandleType(enum.IntEnum,metaclass=_CUexternalMemoryHandleType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipExternalMemoryHandleType
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

HIP_PYTHON_cudaExternalMemoryHandleType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaExternalMemoryHandleType_HALLUCINATE_CONSTANTS","false")

class _cudaExternalMemoryHandleType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaExternalMemoryHandleType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaExternalMemoryHandleType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaExternalMemoryHandleType(enum.IntEnum,metaclass=_cudaExternalMemoryHandleType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipExternalMemoryHandleType
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
CUDA_EXTERNAL_MEMORY_HANDLE_DESC = hip.hip.hipExternalMemoryHandleDesc
CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = hip.hip.hipExternalMemoryHandleDesc
cudaExternalMemoryHandleDesc = hip.hip.hipExternalMemoryHandleDesc
cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(hip.hip.hipExternalMemoryBufferDesc_st):
    pass
CUDA_EXTERNAL_MEMORY_BUFFER_DESC = hip.hip.hipExternalMemoryBufferDesc
CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = hip.hip.hipExternalMemoryBufferDesc
cudaExternalMemoryBufferDesc = hip.hip.hipExternalMemoryBufferDesc

HIP_PYTHON_CUexternalSemaphoreHandleType_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUexternalSemaphoreHandleType_enum_HALLUCINATE_CONSTANTS","false")

class _CUexternalSemaphoreHandleType_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUexternalSemaphoreHandleType_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUexternalSemaphoreHandleType_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUexternalSemaphoreHandleType_enum(enum.IntEnum,metaclass=_CUexternalSemaphoreHandleType_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipExternalSemaphoreHandleType_enum
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

HIP_PYTHON_CUexternalSemaphoreHandleType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUexternalSemaphoreHandleType_HALLUCINATE_CONSTANTS","false")

class _CUexternalSemaphoreHandleType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUexternalSemaphoreHandleType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUexternalSemaphoreHandleType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUexternalSemaphoreHandleType(enum.IntEnum,metaclass=_CUexternalSemaphoreHandleType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipExternalSemaphoreHandleType
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

HIP_PYTHON_cudaExternalSemaphoreHandleType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaExternalSemaphoreHandleType_HALLUCINATE_CONSTANTS","false")

class _cudaExternalSemaphoreHandleType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaExternalSemaphoreHandleType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaExternalSemaphoreHandleType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaExternalSemaphoreHandleType(enum.IntEnum,metaclass=_cudaExternalSemaphoreHandleType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipExternalSemaphoreHandleType
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
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = hip.hip.hipExternalSemaphoreHandleDesc
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = hip.hip.hipExternalSemaphoreHandleDesc
cudaExternalSemaphoreHandleDesc = hip.hip.hipExternalSemaphoreHandleDesc
cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(hip.hip.hipExternalSemaphoreSignalParams_st):
    pass
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = hip.hip.hipExternalSemaphoreSignalParams
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = hip.hip.hipExternalSemaphoreSignalParams
cudaExternalSemaphoreSignalParams = hip.hip.hipExternalSemaphoreSignalParams
cudaExternalSemaphoreSignalParams_v1 = hip.hip.hipExternalSemaphoreSignalParams
cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(hip.hip.hipExternalSemaphoreWaitParams_st):
    pass
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = hip.hip.hipExternalSemaphoreWaitParams
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = hip.hip.hipExternalSemaphoreWaitParams
cudaExternalSemaphoreWaitParams = hip.hip.hipExternalSemaphoreWaitParams
cudaExternalSemaphoreWaitParams_v1 = hip.hip.hipExternalSemaphoreWaitParams

HIP_PYTHON_CUGLDeviceList_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUGLDeviceList_HALLUCINATE_CONSTANTS","false")

class _CUGLDeviceList_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUGLDeviceList_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUGLDeviceList_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUGLDeviceList(enum.IntEnum,metaclass=_CUGLDeviceList_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGLDeviceList
    CU_GL_DEVICE_LIST_ALL = hip.chip.hipGLDeviceListAll
    cudaGLDeviceListAll = hip.chip.hipGLDeviceListAll
    hipGLDeviceListAll = hip.chip.hipGLDeviceListAll
    CU_GL_DEVICE_LIST_CURRENT_FRAME = hip.chip.hipGLDeviceListCurrentFrame
    cudaGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    hipGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    CU_GL_DEVICE_LIST_NEXT_FRAME = hip.chip.hipGLDeviceListNextFrame
    cudaGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame
    hipGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame

HIP_PYTHON_CUGLDeviceList_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUGLDeviceList_enum_HALLUCINATE_CONSTANTS","false")

class _CUGLDeviceList_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUGLDeviceList_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUGLDeviceList_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUGLDeviceList_enum(enum.IntEnum,metaclass=_CUGLDeviceList_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGLDeviceList
    CU_GL_DEVICE_LIST_ALL = hip.chip.hipGLDeviceListAll
    cudaGLDeviceListAll = hip.chip.hipGLDeviceListAll
    hipGLDeviceListAll = hip.chip.hipGLDeviceListAll
    CU_GL_DEVICE_LIST_CURRENT_FRAME = hip.chip.hipGLDeviceListCurrentFrame
    cudaGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    hipGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    CU_GL_DEVICE_LIST_NEXT_FRAME = hip.chip.hipGLDeviceListNextFrame
    cudaGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame
    hipGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame

HIP_PYTHON_cudaGLDeviceList_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGLDeviceList_HALLUCINATE_CONSTANTS","false")

class _cudaGLDeviceList_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGLDeviceList_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGLDeviceList_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaGLDeviceList(enum.IntEnum,metaclass=_cudaGLDeviceList_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGLDeviceList
    CU_GL_DEVICE_LIST_ALL = hip.chip.hipGLDeviceListAll
    cudaGLDeviceListAll = hip.chip.hipGLDeviceListAll
    hipGLDeviceListAll = hip.chip.hipGLDeviceListAll
    CU_GL_DEVICE_LIST_CURRENT_FRAME = hip.chip.hipGLDeviceListCurrentFrame
    cudaGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    hipGLDeviceListCurrentFrame = hip.chip.hipGLDeviceListCurrentFrame
    CU_GL_DEVICE_LIST_NEXT_FRAME = hip.chip.hipGLDeviceListNextFrame
    cudaGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame
    hipGLDeviceListNextFrame = hip.chip.hipGLDeviceListNextFrame

HIP_PYTHON_CUgraphicsRegisterFlags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphicsRegisterFlags_HALLUCINATE_CONSTANTS","false")

class _CUgraphicsRegisterFlags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphicsRegisterFlags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphicsRegisterFlags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphicsRegisterFlags(enum.IntEnum,metaclass=_CUgraphicsRegisterFlags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphicsRegisterFlags
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

HIP_PYTHON_CUgraphicsRegisterFlags_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphicsRegisterFlags_enum_HALLUCINATE_CONSTANTS","false")

class _CUgraphicsRegisterFlags_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphicsRegisterFlags_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphicsRegisterFlags_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphicsRegisterFlags_enum(enum.IntEnum,metaclass=_CUgraphicsRegisterFlags_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphicsRegisterFlags
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

HIP_PYTHON_cudaGraphicsRegisterFlags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphicsRegisterFlags_HALLUCINATE_CONSTANTS","false")

class _cudaGraphicsRegisterFlags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphicsRegisterFlags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphicsRegisterFlags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaGraphicsRegisterFlags(enum.IntEnum,metaclass=_cudaGraphicsRegisterFlags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphicsRegisterFlags
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
CUgraphicsResource_st = hip.hip.hipGraphicsResource
cudaGraphicsResource = hip.hip.hipGraphicsResource
CUgraphicsResource = hip.hip.hipGraphicsResource_t
cudaGraphicsResource_t = hip.hip.hipGraphicsResource_t
cdef class CUgraph_st(hip.hip.ihipGraph):
    pass
CUgraph = hip.hip.hipGraph_t
cudaGraph_t = hip.hip.hipGraph_t
cdef class CUgraphNode_st(hip.hip.hipGraphNode):
    pass
CUgraphNode = hip.hip.hipGraphNode_t
cudaGraphNode_t = hip.hip.hipGraphNode_t
cdef class CUgraphExec_st(hip.hip.hipGraphExec):
    pass
CUgraphExec = hip.hip.hipGraphExec_t
cudaGraphExec_t = hip.hip.hipGraphExec_t
cdef class CUuserObject_st(hip.hip.hipUserObject):
    pass
CUuserObject = hip.hip.hipUserObject_t
cudaUserObject_t = hip.hip.hipUserObject_t

HIP_PYTHON_CUgraphNodeType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphNodeType_HALLUCINATE_CONSTANTS","false")

class _CUgraphNodeType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphNodeType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphNodeType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphNodeType(enum.IntEnum,metaclass=_CUgraphNodeType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphNodeType
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

HIP_PYTHON_CUgraphNodeType_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphNodeType_enum_HALLUCINATE_CONSTANTS","false")

class _CUgraphNodeType_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphNodeType_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphNodeType_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphNodeType_enum(enum.IntEnum,metaclass=_CUgraphNodeType_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphNodeType
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

HIP_PYTHON_cudaGraphNodeType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphNodeType_HALLUCINATE_CONSTANTS","false")

class _cudaGraphNodeType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphNodeType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphNodeType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaGraphNodeType(enum.IntEnum,metaclass=_cudaGraphNodeType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphNodeType
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

HIP_PYTHON_CUkernelNodeAttrID_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUkernelNodeAttrID_HALLUCINATE_CONSTANTS","false")

class _CUkernelNodeAttrID_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUkernelNodeAttrID_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUkernelNodeAttrID_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUkernelNodeAttrID(enum.IntEnum,metaclass=_CUkernelNodeAttrID_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipKernelNodeAttrID
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    cudaKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    hipKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = hip.chip.hipKernelNodeAttributeCooperative
    cudaKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative
    hipKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative

HIP_PYTHON_CUkernelNodeAttrID_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUkernelNodeAttrID_enum_HALLUCINATE_CONSTANTS","false")

class _CUkernelNodeAttrID_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUkernelNodeAttrID_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUkernelNodeAttrID_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUkernelNodeAttrID_enum(enum.IntEnum,metaclass=_CUkernelNodeAttrID_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipKernelNodeAttrID
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    cudaKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    hipKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = hip.chip.hipKernelNodeAttributeCooperative
    cudaKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative
    hipKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative

HIP_PYTHON_cudaKernelNodeAttrID_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaKernelNodeAttrID_HALLUCINATE_CONSTANTS","false")

class _cudaKernelNodeAttrID_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaKernelNodeAttrID_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaKernelNodeAttrID_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaKernelNodeAttrID(enum.IntEnum,metaclass=_cudaKernelNodeAttrID_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipKernelNodeAttrID
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    cudaKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    hipKernelNodeAttributeAccessPolicyWindow = hip.chip.hipKernelNodeAttributeAccessPolicyWindow
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = hip.chip.hipKernelNodeAttributeCooperative
    cudaKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative
    hipKernelNodeAttributeCooperative = hip.chip.hipKernelNodeAttributeCooperative

HIP_PYTHON_CUaccessProperty_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUaccessProperty_HALLUCINATE_CONSTANTS","false")

class _CUaccessProperty_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUaccessProperty_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUaccessProperty_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUaccessProperty(enum.IntEnum,metaclass=_CUaccessProperty_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipAccessProperty
    CU_ACCESS_PROPERTY_NORMAL = hip.chip.hipAccessPropertyNormal
    cudaAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    hipAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    CU_ACCESS_PROPERTY_STREAMING = hip.chip.hipAccessPropertyStreaming
    cudaAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    hipAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    CU_ACCESS_PROPERTY_PERSISTING = hip.chip.hipAccessPropertyPersisting
    cudaAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting
    hipAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting

HIP_PYTHON_CUaccessProperty_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUaccessProperty_enum_HALLUCINATE_CONSTANTS","false")

class _CUaccessProperty_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUaccessProperty_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUaccessProperty_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUaccessProperty_enum(enum.IntEnum,metaclass=_CUaccessProperty_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipAccessProperty
    CU_ACCESS_PROPERTY_NORMAL = hip.chip.hipAccessPropertyNormal
    cudaAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    hipAccessPropertyNormal = hip.chip.hipAccessPropertyNormal
    CU_ACCESS_PROPERTY_STREAMING = hip.chip.hipAccessPropertyStreaming
    cudaAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    hipAccessPropertyStreaming = hip.chip.hipAccessPropertyStreaming
    CU_ACCESS_PROPERTY_PERSISTING = hip.chip.hipAccessPropertyPersisting
    cudaAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting
    hipAccessPropertyPersisting = hip.chip.hipAccessPropertyPersisting

HIP_PYTHON_cudaAccessProperty_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaAccessProperty_HALLUCINATE_CONSTANTS","false")

class _cudaAccessProperty_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaAccessProperty_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaAccessProperty_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaAccessProperty(enum.IntEnum,metaclass=_cudaAccessProperty_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipAccessProperty
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

HIP_PYTHON_CUgraphExecUpdateResult_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphExecUpdateResult_HALLUCINATE_CONSTANTS","false")

class _CUgraphExecUpdateResult_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphExecUpdateResult_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphExecUpdateResult_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphExecUpdateResult(enum.IntEnum,metaclass=_CUgraphExecUpdateResult_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphExecUpdateResult
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

HIP_PYTHON_CUgraphExecUpdateResult_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphExecUpdateResult_enum_HALLUCINATE_CONSTANTS","false")

class _CUgraphExecUpdateResult_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphExecUpdateResult_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphExecUpdateResult_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphExecUpdateResult_enum(enum.IntEnum,metaclass=_CUgraphExecUpdateResult_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphExecUpdateResult
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

HIP_PYTHON_cudaGraphExecUpdateResult_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphExecUpdateResult_HALLUCINATE_CONSTANTS","false")

class _cudaGraphExecUpdateResult_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphExecUpdateResult_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphExecUpdateResult_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaGraphExecUpdateResult(enum.IntEnum,metaclass=_cudaGraphExecUpdateResult_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphExecUpdateResult
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

HIP_PYTHON_CUstreamCaptureMode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamCaptureMode_HALLUCINATE_CONSTANTS","false")

class _CUstreamCaptureMode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamCaptureMode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamCaptureMode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUstreamCaptureMode(enum.IntEnum,metaclass=_CUstreamCaptureMode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipStreamCaptureMode
    CU_STREAM_CAPTURE_MODE_GLOBAL = hip.chip.hipStreamCaptureModeGlobal
    cudaStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    hipStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = hip.chip.hipStreamCaptureModeThreadLocal
    cudaStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    hipStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    CU_STREAM_CAPTURE_MODE_RELAXED = hip.chip.hipStreamCaptureModeRelaxed
    cudaStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed
    hipStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed

HIP_PYTHON_CUstreamCaptureMode_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamCaptureMode_enum_HALLUCINATE_CONSTANTS","false")

class _CUstreamCaptureMode_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamCaptureMode_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamCaptureMode_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUstreamCaptureMode_enum(enum.IntEnum,metaclass=_CUstreamCaptureMode_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipStreamCaptureMode
    CU_STREAM_CAPTURE_MODE_GLOBAL = hip.chip.hipStreamCaptureModeGlobal
    cudaStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    hipStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = hip.chip.hipStreamCaptureModeThreadLocal
    cudaStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    hipStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    CU_STREAM_CAPTURE_MODE_RELAXED = hip.chip.hipStreamCaptureModeRelaxed
    cudaStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed
    hipStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed

HIP_PYTHON_cudaStreamCaptureMode_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaStreamCaptureMode_HALLUCINATE_CONSTANTS","false")

class _cudaStreamCaptureMode_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaStreamCaptureMode_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaStreamCaptureMode_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaStreamCaptureMode(enum.IntEnum,metaclass=_cudaStreamCaptureMode_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipStreamCaptureMode
    CU_STREAM_CAPTURE_MODE_GLOBAL = hip.chip.hipStreamCaptureModeGlobal
    cudaStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    hipStreamCaptureModeGlobal = hip.chip.hipStreamCaptureModeGlobal
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = hip.chip.hipStreamCaptureModeThreadLocal
    cudaStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    hipStreamCaptureModeThreadLocal = hip.chip.hipStreamCaptureModeThreadLocal
    CU_STREAM_CAPTURE_MODE_RELAXED = hip.chip.hipStreamCaptureModeRelaxed
    cudaStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed
    hipStreamCaptureModeRelaxed = hip.chip.hipStreamCaptureModeRelaxed

HIP_PYTHON_CUstreamCaptureStatus_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamCaptureStatus_HALLUCINATE_CONSTANTS","false")

class _CUstreamCaptureStatus_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamCaptureStatus_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamCaptureStatus_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUstreamCaptureStatus(enum.IntEnum,metaclass=_CUstreamCaptureStatus_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipStreamCaptureStatus
    CU_STREAM_CAPTURE_STATUS_NONE = hip.chip.hipStreamCaptureStatusNone
    cudaStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    hipStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    CU_STREAM_CAPTURE_STATUS_ACTIVE = hip.chip.hipStreamCaptureStatusActive
    cudaStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    hipStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = hip.chip.hipStreamCaptureStatusInvalidated
    cudaStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated
    hipStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated

HIP_PYTHON_CUstreamCaptureStatus_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamCaptureStatus_enum_HALLUCINATE_CONSTANTS","false")

class _CUstreamCaptureStatus_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamCaptureStatus_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamCaptureStatus_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUstreamCaptureStatus_enum(enum.IntEnum,metaclass=_CUstreamCaptureStatus_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipStreamCaptureStatus
    CU_STREAM_CAPTURE_STATUS_NONE = hip.chip.hipStreamCaptureStatusNone
    cudaStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    hipStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    CU_STREAM_CAPTURE_STATUS_ACTIVE = hip.chip.hipStreamCaptureStatusActive
    cudaStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    hipStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = hip.chip.hipStreamCaptureStatusInvalidated
    cudaStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated
    hipStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated

HIP_PYTHON_cudaStreamCaptureStatus_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaStreamCaptureStatus_HALLUCINATE_CONSTANTS","false")

class _cudaStreamCaptureStatus_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaStreamCaptureStatus_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaStreamCaptureStatus_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaStreamCaptureStatus(enum.IntEnum,metaclass=_cudaStreamCaptureStatus_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipStreamCaptureStatus
    CU_STREAM_CAPTURE_STATUS_NONE = hip.chip.hipStreamCaptureStatusNone
    cudaStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    hipStreamCaptureStatusNone = hip.chip.hipStreamCaptureStatusNone
    CU_STREAM_CAPTURE_STATUS_ACTIVE = hip.chip.hipStreamCaptureStatusActive
    cudaStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    hipStreamCaptureStatusActive = hip.chip.hipStreamCaptureStatusActive
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = hip.chip.hipStreamCaptureStatusInvalidated
    cudaStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated
    hipStreamCaptureStatusInvalidated = hip.chip.hipStreamCaptureStatusInvalidated

HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_HALLUCINATE_CONSTANTS","false")

class _CUstreamUpdateCaptureDependencies_flags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUstreamUpdateCaptureDependencies_flags(enum.IntEnum,metaclass=_CUstreamUpdateCaptureDependencies_flags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipStreamUpdateCaptureDependenciesFlags
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = hip.chip.hipStreamAddCaptureDependencies
    cudaStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    hipStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = hip.chip.hipStreamSetCaptureDependencies
    cudaStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies
    hipStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies

HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_enum_HALLUCINATE_CONSTANTS","false")

class _CUstreamUpdateCaptureDependencies_flags_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUstreamUpdateCaptureDependencies_flags_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUstreamUpdateCaptureDependencies_flags_enum(enum.IntEnum,metaclass=_CUstreamUpdateCaptureDependencies_flags_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipStreamUpdateCaptureDependenciesFlags
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = hip.chip.hipStreamAddCaptureDependencies
    cudaStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    hipStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = hip.chip.hipStreamSetCaptureDependencies
    cudaStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies
    hipStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies

HIP_PYTHON_cudaStreamUpdateCaptureDependenciesFlags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaStreamUpdateCaptureDependenciesFlags_HALLUCINATE_CONSTANTS","false")

class _cudaStreamUpdateCaptureDependenciesFlags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaStreamUpdateCaptureDependenciesFlags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaStreamUpdateCaptureDependenciesFlags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaStreamUpdateCaptureDependenciesFlags(enum.IntEnum,metaclass=_cudaStreamUpdateCaptureDependenciesFlags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipStreamUpdateCaptureDependenciesFlags
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = hip.chip.hipStreamAddCaptureDependencies
    cudaStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    hipStreamAddCaptureDependencies = hip.chip.hipStreamAddCaptureDependencies
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = hip.chip.hipStreamSetCaptureDependencies
    cudaStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies
    hipStreamSetCaptureDependencies = hip.chip.hipStreamSetCaptureDependencies

HIP_PYTHON_CUgraphMem_attribute_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphMem_attribute_HALLUCINATE_CONSTANTS","false")

class _CUgraphMem_attribute_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphMem_attribute_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphMem_attribute_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphMem_attribute(enum.IntEnum,metaclass=_CUgraphMem_attribute_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphMemAttributeType
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

HIP_PYTHON_CUgraphMem_attribute_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphMem_attribute_enum_HALLUCINATE_CONSTANTS","false")

class _CUgraphMem_attribute_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphMem_attribute_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphMem_attribute_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphMem_attribute_enum(enum.IntEnum,metaclass=_CUgraphMem_attribute_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphMemAttributeType
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

HIP_PYTHON_cudaGraphMemAttributeType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphMemAttributeType_HALLUCINATE_CONSTANTS","false")

class _cudaGraphMemAttributeType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphMemAttributeType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphMemAttributeType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaGraphMemAttributeType(enum.IntEnum,metaclass=_cudaGraphMemAttributeType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphMemAttributeType
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

HIP_PYTHON_CUuserObject_flags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUuserObject_flags_HALLUCINATE_CONSTANTS","false")

class _CUuserObject_flags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUuserObject_flags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUuserObject_flags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUuserObject_flags(enum.IntEnum,metaclass=_CUuserObject_flags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipUserObjectFlags
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = hip.chip.hipUserObjectNoDestructorSync
    cudaUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync
    hipUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync

HIP_PYTHON_CUuserObject_flags_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUuserObject_flags_enum_HALLUCINATE_CONSTANTS","false")

class _CUuserObject_flags_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUuserObject_flags_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUuserObject_flags_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUuserObject_flags_enum(enum.IntEnum,metaclass=_CUuserObject_flags_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipUserObjectFlags
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = hip.chip.hipUserObjectNoDestructorSync
    cudaUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync
    hipUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync

HIP_PYTHON_cudaUserObjectFlags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaUserObjectFlags_HALLUCINATE_CONSTANTS","false")

class _cudaUserObjectFlags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaUserObjectFlags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaUserObjectFlags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaUserObjectFlags(enum.IntEnum,metaclass=_cudaUserObjectFlags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipUserObjectFlags
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = hip.chip.hipUserObjectNoDestructorSync
    cudaUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync
    hipUserObjectNoDestructorSync = hip.chip.hipUserObjectNoDestructorSync

HIP_PYTHON_CUuserObjectRetain_flags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUuserObjectRetain_flags_HALLUCINATE_CONSTANTS","false")

class _CUuserObjectRetain_flags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUuserObjectRetain_flags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUuserObjectRetain_flags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUuserObjectRetain_flags(enum.IntEnum,metaclass=_CUuserObjectRetain_flags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipUserObjectRetainFlags
    CU_GRAPH_USER_OBJECT_MOVE = hip.chip.hipGraphUserObjectMove
    cudaGraphUserObjectMove = hip.chip.hipGraphUserObjectMove
    hipGraphUserObjectMove = hip.chip.hipGraphUserObjectMove

HIP_PYTHON_CUuserObjectRetain_flags_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUuserObjectRetain_flags_enum_HALLUCINATE_CONSTANTS","false")

class _CUuserObjectRetain_flags_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUuserObjectRetain_flags_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUuserObjectRetain_flags_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUuserObjectRetain_flags_enum(enum.IntEnum,metaclass=_CUuserObjectRetain_flags_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipUserObjectRetainFlags
    CU_GRAPH_USER_OBJECT_MOVE = hip.chip.hipGraphUserObjectMove
    cudaGraphUserObjectMove = hip.chip.hipGraphUserObjectMove
    hipGraphUserObjectMove = hip.chip.hipGraphUserObjectMove

HIP_PYTHON_cudaUserObjectRetainFlags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaUserObjectRetainFlags_HALLUCINATE_CONSTANTS","false")

class _cudaUserObjectRetainFlags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaUserObjectRetainFlags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaUserObjectRetainFlags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaUserObjectRetainFlags(enum.IntEnum,metaclass=_cudaUserObjectRetainFlags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipUserObjectRetainFlags
    CU_GRAPH_USER_OBJECT_MOVE = hip.chip.hipGraphUserObjectMove
    cudaGraphUserObjectMove = hip.chip.hipGraphUserObjectMove
    hipGraphUserObjectMove = hip.chip.hipGraphUserObjectMove

HIP_PYTHON_CUgraphInstantiate_flags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphInstantiate_flags_HALLUCINATE_CONSTANTS","false")

class _CUgraphInstantiate_flags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphInstantiate_flags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphInstantiate_flags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphInstantiate_flags(enum.IntEnum,metaclass=_CUgraphInstantiate_flags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphInstantiateFlags
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    cudaGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    hipGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch

HIP_PYTHON_CUgraphInstantiate_flags_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUgraphInstantiate_flags_enum_HALLUCINATE_CONSTANTS","false")

class _CUgraphInstantiate_flags_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUgraphInstantiate_flags_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUgraphInstantiate_flags_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUgraphInstantiate_flags_enum(enum.IntEnum,metaclass=_CUgraphInstantiate_flags_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphInstantiateFlags
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    cudaGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    hipGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch

HIP_PYTHON_cudaGraphInstantiateFlags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_cudaGraphInstantiateFlags_HALLUCINATE_CONSTANTS","false")

class _cudaGraphInstantiateFlags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_cudaGraphInstantiateFlags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_cudaGraphInstantiateFlags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class cudaGraphInstantiateFlags(enum.IntEnum,metaclass=_cudaGraphInstantiateFlags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipGraphInstantiateFlags
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    cudaGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
    hipGraphInstantiateFlagAutoFreeOnLaunch = hip.chip.hipGraphInstantiateFlagAutoFreeOnLaunch
cdef class CUmemAllocationProp(hip.hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_st(hip.hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_v1(hip.hip.hipMemAllocationProp):
    pass
CUmemGenericAllocationHandle = hip.hip.hipMemGenericAllocationHandle_t
CUmemGenericAllocationHandle_v1 = hip.hip.hipMemGenericAllocationHandle_t

HIP_PYTHON_CUmemAllocationGranularity_flags_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationGranularity_flags_HALLUCINATE_CONSTANTS","false")

class _CUmemAllocationGranularity_flags_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationGranularity_flags_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationGranularity_flags_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemAllocationGranularity_flags(enum.IntEnum,metaclass=_CUmemAllocationGranularity_flags_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAllocationGranularity_flags
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = hip.chip.hipMemAllocationGranularityMinimum
    hipMemAllocationGranularityMinimum = hip.chip.hipMemAllocationGranularityMinimum
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = hip.chip.hipMemAllocationGranularityRecommended
    hipMemAllocationGranularityRecommended = hip.chip.hipMemAllocationGranularityRecommended

HIP_PYTHON_CUmemAllocationGranularity_flags_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemAllocationGranularity_flags_enum_HALLUCINATE_CONSTANTS","false")

class _CUmemAllocationGranularity_flags_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemAllocationGranularity_flags_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemAllocationGranularity_flags_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemAllocationGranularity_flags_enum(enum.IntEnum,metaclass=_CUmemAllocationGranularity_flags_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemAllocationGranularity_flags
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = hip.chip.hipMemAllocationGranularityMinimum
    hipMemAllocationGranularityMinimum = hip.chip.hipMemAllocationGranularityMinimum
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = hip.chip.hipMemAllocationGranularityRecommended
    hipMemAllocationGranularityRecommended = hip.chip.hipMemAllocationGranularityRecommended

HIP_PYTHON_CUmemHandleType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemHandleType_HALLUCINATE_CONSTANTS","false")

class _CUmemHandleType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemHandleType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemHandleType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemHandleType(enum.IntEnum,metaclass=_CUmemHandleType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemHandleType
    CU_MEM_HANDLE_TYPE_GENERIC = hip.chip.hipMemHandleTypeGeneric
    hipMemHandleTypeGeneric = hip.chip.hipMemHandleTypeGeneric

HIP_PYTHON_CUmemHandleType_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemHandleType_enum_HALLUCINATE_CONSTANTS","false")

class _CUmemHandleType_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemHandleType_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemHandleType_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemHandleType_enum(enum.IntEnum,metaclass=_CUmemHandleType_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemHandleType
    CU_MEM_HANDLE_TYPE_GENERIC = hip.chip.hipMemHandleTypeGeneric
    hipMemHandleTypeGeneric = hip.chip.hipMemHandleTypeGeneric

HIP_PYTHON_CUmemOperationType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemOperationType_HALLUCINATE_CONSTANTS","false")

class _CUmemOperationType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemOperationType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemOperationType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemOperationType(enum.IntEnum,metaclass=_CUmemOperationType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemOperationType
    CU_MEM_OPERATION_TYPE_MAP = hip.chip.hipMemOperationTypeMap
    hipMemOperationTypeMap = hip.chip.hipMemOperationTypeMap
    CU_MEM_OPERATION_TYPE_UNMAP = hip.chip.hipMemOperationTypeUnmap
    hipMemOperationTypeUnmap = hip.chip.hipMemOperationTypeUnmap

HIP_PYTHON_CUmemOperationType_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUmemOperationType_enum_HALLUCINATE_CONSTANTS","false")

class _CUmemOperationType_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUmemOperationType_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUmemOperationType_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUmemOperationType_enum(enum.IntEnum,metaclass=_CUmemOperationType_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipMemOperationType
    CU_MEM_OPERATION_TYPE_MAP = hip.chip.hipMemOperationTypeMap
    hipMemOperationTypeMap = hip.chip.hipMemOperationTypeMap
    CU_MEM_OPERATION_TYPE_UNMAP = hip.chip.hipMemOperationTypeUnmap
    hipMemOperationTypeUnmap = hip.chip.hipMemOperationTypeUnmap

HIP_PYTHON_CUarraySparseSubresourceType_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUarraySparseSubresourceType_HALLUCINATE_CONSTANTS","false")

class _CUarraySparseSubresourceType_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUarraySparseSubresourceType_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUarraySparseSubresourceType_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUarraySparseSubresourceType(enum.IntEnum,metaclass=_CUarraySparseSubresourceType_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipArraySparseSubresourceType
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = hip.chip.hipArraySparseSubresourceTypeSparseLevel
    hipArraySparseSubresourceTypeSparseLevel = hip.chip.hipArraySparseSubresourceTypeSparseLevel
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = hip.chip.hipArraySparseSubresourceTypeMiptail
    hipArraySparseSubresourceTypeMiptail = hip.chip.hipArraySparseSubresourceTypeMiptail

HIP_PYTHON_CUarraySparseSubresourceType_enum_HALLUCINATE_CONSTANTS = _hip_python_get_bool_environ_var("HIP_PYTHON_CUarraySparseSubresourceType_enum_HALLUCINATE_CONSTANTS","false")

class _CUarraySparseSubresourceType_enum_EnumMeta(enum.EnumMeta):

    class FakeEnumType():
        """Mimicks the orginal enum type this 
        is derived from.
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
            if isinstance(other,self._orig_enum_type_):
                return self.value == other.value
            return False

        @property
        def __class__(self):
            """Overwrite __class__ to satisfy __isinstance__ check.
            """
            return self._orig_enum_type_

        def __repr__(self):        
            """Mimicks enum.Enum.__repr__"""
            return "<%s.%s: %r>" % (
                    self.__class__.__name__, self._name_, self._value_)

        def __str__(self):
            """Mimicks enum.Enum.__str__"""
            return "%s.%s" % (self.__class__.__name__, self._name_)

        def __hash__(self):
            return hash(str(self))

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUarraySparseSubresourceType_enum_HALLUCINATE_CONSTANTS
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUarraySparseSubresourceType_enum_HALLUCINATE_CONSTANTS:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                fake_enum = type(
                    name,
                    (cls.FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return fake_enum


class CUarraySparseSubresourceType_enum(enum.IntEnum,metaclass=_CUarraySparseSubresourceType_enum_EnumMeta):
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return hip.hip.hipArraySparseSubresourceType
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
cuInit = hip.hip.hipInit
cuDriverGetVersion = hip.hip.hipDriverGetVersion
cudaDriverGetVersion = hip.hip.hipDriverGetVersion
cudaRuntimeGetVersion = hip.hip.hipRuntimeGetVersion
cuDeviceGet = hip.hip.hipDeviceGet
cuDeviceComputeCapability = hip.hip.hipDeviceComputeCapability
cuDeviceGetName = hip.hip.hipDeviceGetName
cuDeviceGetUuid = hip.hip.hipDeviceGetUuid
cuDeviceGetUuid_v2 = hip.hip.hipDeviceGetUuid
cudaDeviceGetP2PAttribute = hip.hip.hipDeviceGetP2PAttribute
cuDeviceGetP2PAttribute = hip.hip.hipDeviceGetP2PAttribute
cudaDeviceGetPCIBusId = hip.hip.hipDeviceGetPCIBusId
cuDeviceGetPCIBusId = hip.hip.hipDeviceGetPCIBusId
cudaDeviceGetByPCIBusId = hip.hip.hipDeviceGetByPCIBusId
cuDeviceGetByPCIBusId = hip.hip.hipDeviceGetByPCIBusId
cuDeviceTotalMem = hip.hip.hipDeviceTotalMem
cuDeviceTotalMem_v2 = hip.hip.hipDeviceTotalMem
cudaDeviceSynchronize = hip.hip.hipDeviceSynchronize
cudaThreadSynchronize = hip.hip.hipDeviceSynchronize
cudaDeviceReset = hip.hip.hipDeviceReset
cudaThreadExit = hip.hip.hipDeviceReset
cudaSetDevice = hip.hip.hipSetDevice
cudaGetDevice = hip.hip.hipGetDevice
cuDeviceGetCount = hip.hip.hipGetDeviceCount
cudaGetDeviceCount = hip.hip.hipGetDeviceCount
cuDeviceGetAttribute = hip.hip.hipDeviceGetAttribute
cudaDeviceGetAttribute = hip.hip.hipDeviceGetAttribute
cuDeviceGetDefaultMemPool = hip.hip.hipDeviceGetDefaultMemPool
cudaDeviceGetDefaultMemPool = hip.hip.hipDeviceGetDefaultMemPool
cuDeviceSetMemPool = hip.hip.hipDeviceSetMemPool
cudaDeviceSetMemPool = hip.hip.hipDeviceSetMemPool
cuDeviceGetMemPool = hip.hip.hipDeviceGetMemPool
cudaDeviceGetMemPool = hip.hip.hipDeviceGetMemPool
cudaGetDeviceProperties = hip.hip.hipGetDeviceProperties
cudaDeviceSetCacheConfig = hip.hip.hipDeviceSetCacheConfig
cudaThreadSetCacheConfig = hip.hip.hipDeviceSetCacheConfig
cudaDeviceGetCacheConfig = hip.hip.hipDeviceGetCacheConfig
cudaThreadGetCacheConfig = hip.hip.hipDeviceGetCacheConfig
cudaDeviceGetLimit = hip.hip.hipDeviceGetLimit
cuCtxGetLimit = hip.hip.hipDeviceGetLimit
cudaDeviceSetLimit = hip.hip.hipDeviceSetLimit
cuCtxSetLimit = hip.hip.hipDeviceSetLimit
cudaDeviceGetSharedMemConfig = hip.hip.hipDeviceGetSharedMemConfig
cudaGetDeviceFlags = hip.hip.hipGetDeviceFlags
cudaDeviceSetSharedMemConfig = hip.hip.hipDeviceSetSharedMemConfig
cudaSetDeviceFlags = hip.hip.hipSetDeviceFlags
cudaChooseDevice = hip.hip.hipChooseDevice
cudaIpcGetMemHandle = hip.hip.hipIpcGetMemHandle
cuIpcGetMemHandle = hip.hip.hipIpcGetMemHandle
cudaIpcOpenMemHandle = hip.hip.hipIpcOpenMemHandle
cuIpcOpenMemHandle = hip.hip.hipIpcOpenMemHandle
cudaIpcCloseMemHandle = hip.hip.hipIpcCloseMemHandle
cuIpcCloseMemHandle = hip.hip.hipIpcCloseMemHandle
cudaIpcGetEventHandle = hip.hip.hipIpcGetEventHandle
cuIpcGetEventHandle = hip.hip.hipIpcGetEventHandle
cudaIpcOpenEventHandle = hip.hip.hipIpcOpenEventHandle
cuIpcOpenEventHandle = hip.hip.hipIpcOpenEventHandle
cudaFuncSetAttribute = hip.hip.hipFuncSetAttribute
cudaFuncSetCacheConfig = hip.hip.hipFuncSetCacheConfig
cudaFuncSetSharedMemConfig = hip.hip.hipFuncSetSharedMemConfig
cudaGetLastError = hip.hip.hipGetLastError
cudaPeekAtLastError = hip.hip.hipPeekAtLastError
cudaGetErrorName = hip.hip.hipGetErrorName
cudaGetErrorString = hip.hip.hipGetErrorString
cuGetErrorName = hip.hip.hipDrvGetErrorName
cuGetErrorString = hip.hip.hipDrvGetErrorString
cudaStreamCreate = hip.hip.hipStreamCreate
cuStreamCreate = hip.hip.hipStreamCreateWithFlags
cudaStreamCreateWithFlags = hip.hip.hipStreamCreateWithFlags
cuStreamCreateWithPriority = hip.hip.hipStreamCreateWithPriority
cudaStreamCreateWithPriority = hip.hip.hipStreamCreateWithPriority
cudaDeviceGetStreamPriorityRange = hip.hip.hipDeviceGetStreamPriorityRange
cuCtxGetStreamPriorityRange = hip.hip.hipDeviceGetStreamPriorityRange
cuStreamDestroy = hip.hip.hipStreamDestroy
cuStreamDestroy_v2 = hip.hip.hipStreamDestroy
cudaStreamDestroy = hip.hip.hipStreamDestroy
cuStreamQuery = hip.hip.hipStreamQuery
cudaStreamQuery = hip.hip.hipStreamQuery
cuStreamSynchronize = hip.hip.hipStreamSynchronize
cudaStreamSynchronize = hip.hip.hipStreamSynchronize
cuStreamWaitEvent = hip.hip.hipStreamWaitEvent
cudaStreamWaitEvent = hip.hip.hipStreamWaitEvent
cuStreamGetFlags = hip.hip.hipStreamGetFlags
cudaStreamGetFlags = hip.hip.hipStreamGetFlags
cuStreamGetPriority = hip.hip.hipStreamGetPriority
cudaStreamGetPriority = hip.hip.hipStreamGetPriority
cdef class CUstreamCallback(hip.hip.hipStreamCallback_t):
    pass
cdef class cudaStreamCallback_t(hip.hip.hipStreamCallback_t):
    pass
cuStreamAddCallback = hip.hip.hipStreamAddCallback
cudaStreamAddCallback = hip.hip.hipStreamAddCallback
cuStreamWaitValue32 = hip.hip.hipStreamWaitValue32
cuStreamWaitValue32_v2 = hip.hip.hipStreamWaitValue32
cuStreamWaitValue64 = hip.hip.hipStreamWaitValue64
cuStreamWaitValue64_v2 = hip.hip.hipStreamWaitValue64
cuStreamWriteValue32 = hip.hip.hipStreamWriteValue32
cuStreamWriteValue32_v2 = hip.hip.hipStreamWriteValue32
cuStreamWriteValue64 = hip.hip.hipStreamWriteValue64
cuStreamWriteValue64_v2 = hip.hip.hipStreamWriteValue64
cuEventCreate = hip.hip.hipEventCreateWithFlags
cudaEventCreateWithFlags = hip.hip.hipEventCreateWithFlags
cudaEventCreate = hip.hip.hipEventCreate
cuEventRecord = hip.hip.hipEventRecord
cudaEventRecord = hip.hip.hipEventRecord
cuEventDestroy = hip.hip.hipEventDestroy
cuEventDestroy_v2 = hip.hip.hipEventDestroy
cudaEventDestroy = hip.hip.hipEventDestroy
cuEventSynchronize = hip.hip.hipEventSynchronize
cudaEventSynchronize = hip.hip.hipEventSynchronize
cuEventElapsedTime = hip.hip.hipEventElapsedTime
cudaEventElapsedTime = hip.hip.hipEventElapsedTime
cuEventQuery = hip.hip.hipEventQuery
cudaEventQuery = hip.hip.hipEventQuery
cudaPointerGetAttributes = hip.hip.hipPointerGetAttributes
cuPointerGetAttribute = hip.hip.hipPointerGetAttribute
cuPointerGetAttributes = hip.hip.hipDrvPointerGetAttributes
cuImportExternalSemaphore = hip.hip.hipImportExternalSemaphore
cudaImportExternalSemaphore = hip.hip.hipImportExternalSemaphore
cuSignalExternalSemaphoresAsync = hip.hip.hipSignalExternalSemaphoresAsync
cudaSignalExternalSemaphoresAsync = hip.hip.hipSignalExternalSemaphoresAsync
cuWaitExternalSemaphoresAsync = hip.hip.hipWaitExternalSemaphoresAsync
cudaWaitExternalSemaphoresAsync = hip.hip.hipWaitExternalSemaphoresAsync
cuDestroyExternalSemaphore = hip.hip.hipDestroyExternalSemaphore
cudaDestroyExternalSemaphore = hip.hip.hipDestroyExternalSemaphore
cuImportExternalMemory = hip.hip.hipImportExternalMemory
cudaImportExternalMemory = hip.hip.hipImportExternalMemory
cuExternalMemoryGetMappedBuffer = hip.hip.hipExternalMemoryGetMappedBuffer
cudaExternalMemoryGetMappedBuffer = hip.hip.hipExternalMemoryGetMappedBuffer
cuDestroyExternalMemory = hip.hip.hipDestroyExternalMemory
cudaDestroyExternalMemory = hip.hip.hipDestroyExternalMemory
cuMemAlloc = hip.hip.hipMalloc
cuMemAlloc_v2 = hip.hip.hipMalloc
cudaMalloc = hip.hip.hipMalloc
cuMemAllocHost = hip.hip.hipMemAllocHost
cuMemAllocHost_v2 = hip.hip.hipMemAllocHost
cudaMallocHost = hip.hip.hipHostMalloc
cuMemAllocManaged = hip.hip.hipMallocManaged
cudaMallocManaged = hip.hip.hipMallocManaged
cudaMemPrefetchAsync = hip.hip.hipMemPrefetchAsync
cuMemPrefetchAsync = hip.hip.hipMemPrefetchAsync
cudaMemAdvise = hip.hip.hipMemAdvise
cuMemAdvise = hip.hip.hipMemAdvise
cudaMemRangeGetAttribute = hip.hip.hipMemRangeGetAttribute
cuMemRangeGetAttribute = hip.hip.hipMemRangeGetAttribute
cudaMemRangeGetAttributes = hip.hip.hipMemRangeGetAttributes
cuMemRangeGetAttributes = hip.hip.hipMemRangeGetAttributes
cuStreamAttachMemAsync = hip.hip.hipStreamAttachMemAsync
cudaStreamAttachMemAsync = hip.hip.hipStreamAttachMemAsync
cudaMallocAsync = hip.hip.hipMallocAsync
cuMemAllocAsync = hip.hip.hipMallocAsync
cudaFreeAsync = hip.hip.hipFreeAsync
cuMemFreeAsync = hip.hip.hipFreeAsync
cudaMemPoolTrimTo = hip.hip.hipMemPoolTrimTo
cuMemPoolTrimTo = hip.hip.hipMemPoolTrimTo
cudaMemPoolSetAttribute = hip.hip.hipMemPoolSetAttribute
cuMemPoolSetAttribute = hip.hip.hipMemPoolSetAttribute
cudaMemPoolGetAttribute = hip.hip.hipMemPoolGetAttribute
cuMemPoolGetAttribute = hip.hip.hipMemPoolGetAttribute
cudaMemPoolSetAccess = hip.hip.hipMemPoolSetAccess
cuMemPoolSetAccess = hip.hip.hipMemPoolSetAccess
cudaMemPoolGetAccess = hip.hip.hipMemPoolGetAccess
cuMemPoolGetAccess = hip.hip.hipMemPoolGetAccess
cudaMemPoolCreate = hip.hip.hipMemPoolCreate
cuMemPoolCreate = hip.hip.hipMemPoolCreate
cudaMemPoolDestroy = hip.hip.hipMemPoolDestroy
cuMemPoolDestroy = hip.hip.hipMemPoolDestroy
cudaMallocFromPoolAsync = hip.hip.hipMallocFromPoolAsync
cuMemAllocFromPoolAsync = hip.hip.hipMallocFromPoolAsync
cudaMemPoolExportToShareableHandle = hip.hip.hipMemPoolExportToShareableHandle
cuMemPoolExportToShareableHandle = hip.hip.hipMemPoolExportToShareableHandle
cudaMemPoolImportFromShareableHandle = hip.hip.hipMemPoolImportFromShareableHandle
cuMemPoolImportFromShareableHandle = hip.hip.hipMemPoolImportFromShareableHandle
cudaMemPoolExportPointer = hip.hip.hipMemPoolExportPointer
cuMemPoolExportPointer = hip.hip.hipMemPoolExportPointer
cudaMemPoolImportPointer = hip.hip.hipMemPoolImportPointer
cuMemPoolImportPointer = hip.hip.hipMemPoolImportPointer
cuMemHostAlloc = hip.hip.hipHostAlloc
cudaHostAlloc = hip.hip.hipHostAlloc
cuMemHostGetDevicePointer = hip.hip.hipHostGetDevicePointer
cuMemHostGetDevicePointer_v2 = hip.hip.hipHostGetDevicePointer
cudaHostGetDevicePointer = hip.hip.hipHostGetDevicePointer
cuMemHostGetFlags = hip.hip.hipHostGetFlags
cudaHostGetFlags = hip.hip.hipHostGetFlags
cuMemHostRegister = hip.hip.hipHostRegister
cuMemHostRegister_v2 = hip.hip.hipHostRegister
cudaHostRegister = hip.hip.hipHostRegister
cuMemHostUnregister = hip.hip.hipHostUnregister
cudaHostUnregister = hip.hip.hipHostUnregister
cudaMallocPitch = hip.hip.hipMallocPitch
cuMemAllocPitch = hip.hip.hipMemAllocPitch
cuMemAllocPitch_v2 = hip.hip.hipMemAllocPitch
cuMemFree = hip.hip.hipFree
cuMemFree_v2 = hip.hip.hipFree
cudaFree = hip.hip.hipFree
cuMemFreeHost = hip.hip.hipHostFree
cudaFreeHost = hip.hip.hipHostFree
cudaMemcpy = hip.hip.hipMemcpy
cuMemcpyHtoD = hip.hip.hipMemcpyHtoD
cuMemcpyHtoD_v2 = hip.hip.hipMemcpyHtoD
cuMemcpyDtoH = hip.hip.hipMemcpyDtoH
cuMemcpyDtoH_v2 = hip.hip.hipMemcpyDtoH
cuMemcpyDtoD = hip.hip.hipMemcpyDtoD
cuMemcpyDtoD_v2 = hip.hip.hipMemcpyDtoD
cuMemcpyHtoDAsync = hip.hip.hipMemcpyHtoDAsync
cuMemcpyHtoDAsync_v2 = hip.hip.hipMemcpyHtoDAsync
cuMemcpyDtoHAsync = hip.hip.hipMemcpyDtoHAsync
cuMemcpyDtoHAsync_v2 = hip.hip.hipMemcpyDtoHAsync
cuMemcpyDtoDAsync = hip.hip.hipMemcpyDtoDAsync
cuMemcpyDtoDAsync_v2 = hip.hip.hipMemcpyDtoDAsync
cuModuleGetGlobal = hip.hip.hipModuleGetGlobal
cuModuleGetGlobal_v2 = hip.hip.hipModuleGetGlobal
cudaGetSymbolAddress = hip.hip.hipGetSymbolAddress
cudaGetSymbolSize = hip.hip.hipGetSymbolSize
cudaMemcpyToSymbol = hip.hip.hipMemcpyToSymbol
cudaMemcpyToSymbolAsync = hip.hip.hipMemcpyToSymbolAsync
cudaMemcpyFromSymbol = hip.hip.hipMemcpyFromSymbol
cudaMemcpyFromSymbolAsync = hip.hip.hipMemcpyFromSymbolAsync
cudaMemcpyAsync = hip.hip.hipMemcpyAsync
cudaMemset = hip.hip.hipMemset
cuMemsetD8 = hip.hip.hipMemsetD8
cuMemsetD8_v2 = hip.hip.hipMemsetD8
cuMemsetD8Async = hip.hip.hipMemsetD8Async
cuMemsetD16 = hip.hip.hipMemsetD16
cuMemsetD16_v2 = hip.hip.hipMemsetD16
cuMemsetD16Async = hip.hip.hipMemsetD16Async
cuMemsetD32 = hip.hip.hipMemsetD32
cuMemsetD32_v2 = hip.hip.hipMemsetD32
cudaMemsetAsync = hip.hip.hipMemsetAsync
cuMemsetD32Async = hip.hip.hipMemsetD32Async
cudaMemset2D = hip.hip.hipMemset2D
cudaMemset2DAsync = hip.hip.hipMemset2DAsync
cudaMemset3D = hip.hip.hipMemset3D
cudaMemset3DAsync = hip.hip.hipMemset3DAsync
cuMemGetInfo = hip.hip.hipMemGetInfo
cuMemGetInfo_v2 = hip.hip.hipMemGetInfo
cudaMemGetInfo = hip.hip.hipMemGetInfo
cudaMallocArray = hip.hip.hipMallocArray
cuArrayCreate = hip.hip.hipArrayCreate
cuArrayCreate_v2 = hip.hip.hipArrayCreate
cuArrayDestroy = hip.hip.hipArrayDestroy
cuArray3DCreate = hip.hip.hipArray3DCreate
cuArray3DCreate_v2 = hip.hip.hipArray3DCreate
cudaMalloc3D = hip.hip.hipMalloc3D
cudaFreeArray = hip.hip.hipFreeArray
cudaFreeMipmappedArray = hip.hip.hipFreeMipmappedArray
cudaMalloc3DArray = hip.hip.hipMalloc3DArray
cudaMallocMipmappedArray = hip.hip.hipMallocMipmappedArray
cudaGetMipmappedArrayLevel = hip.hip.hipGetMipmappedArrayLevel
cudaMemcpy2D = hip.hip.hipMemcpy2D
cuMemcpy2D = hip.hip.hipMemcpyParam2D
cuMemcpy2D_v2 = hip.hip.hipMemcpyParam2D
cuMemcpy2DAsync = hip.hip.hipMemcpyParam2DAsync
cuMemcpy2DAsync_v2 = hip.hip.hipMemcpyParam2DAsync
cudaMemcpy2DAsync = hip.hip.hipMemcpy2DAsync
cudaMemcpy2DToArray = hip.hip.hipMemcpy2DToArray
cudaMemcpy2DToArrayAsync = hip.hip.hipMemcpy2DToArrayAsync
cudaMemcpyToArray = hip.hip.hipMemcpyToArray
cudaMemcpyFromArray = hip.hip.hipMemcpyFromArray
cudaMemcpy2DFromArray = hip.hip.hipMemcpy2DFromArray
cudaMemcpy2DFromArrayAsync = hip.hip.hipMemcpy2DFromArrayAsync
cuMemcpyAtoH = hip.hip.hipMemcpyAtoH
cuMemcpyAtoH_v2 = hip.hip.hipMemcpyAtoH
cuMemcpyHtoA = hip.hip.hipMemcpyHtoA
cuMemcpyHtoA_v2 = hip.hip.hipMemcpyHtoA
cudaMemcpy3D = hip.hip.hipMemcpy3D
cudaMemcpy3DAsync = hip.hip.hipMemcpy3DAsync
cuMemcpy3D = hip.hip.hipDrvMemcpy3D
cuMemcpy3D_v2 = hip.hip.hipDrvMemcpy3D
cuMemcpy3DAsync = hip.hip.hipDrvMemcpy3DAsync
cuMemcpy3DAsync_v2 = hip.hip.hipDrvMemcpy3DAsync
cuDeviceCanAccessPeer = hip.hip.hipDeviceCanAccessPeer
cudaDeviceCanAccessPeer = hip.hip.hipDeviceCanAccessPeer
cudaDeviceEnablePeerAccess = hip.hip.hipDeviceEnablePeerAccess
cudaDeviceDisablePeerAccess = hip.hip.hipDeviceDisablePeerAccess
cuMemGetAddressRange = hip.hip.hipMemGetAddressRange
cuMemGetAddressRange_v2 = hip.hip.hipMemGetAddressRange
cudaMemcpyPeer = hip.hip.hipMemcpyPeer
cudaMemcpyPeerAsync = hip.hip.hipMemcpyPeerAsync
cuCtxCreate = hip.hip.hipCtxCreate
cuCtxCreate_v2 = hip.hip.hipCtxCreate
cuCtxDestroy = hip.hip.hipCtxDestroy
cuCtxDestroy_v2 = hip.hip.hipCtxDestroy
cuCtxPopCurrent = hip.hip.hipCtxPopCurrent
cuCtxPopCurrent_v2 = hip.hip.hipCtxPopCurrent
cuCtxPushCurrent = hip.hip.hipCtxPushCurrent
cuCtxPushCurrent_v2 = hip.hip.hipCtxPushCurrent
cuCtxSetCurrent = hip.hip.hipCtxSetCurrent
cuCtxGetCurrent = hip.hip.hipCtxGetCurrent
cuCtxGetDevice = hip.hip.hipCtxGetDevice
cuCtxGetApiVersion = hip.hip.hipCtxGetApiVersion
cuCtxGetCacheConfig = hip.hip.hipCtxGetCacheConfig
cuCtxSetCacheConfig = hip.hip.hipCtxSetCacheConfig
cuCtxSetSharedMemConfig = hip.hip.hipCtxSetSharedMemConfig
cuCtxGetSharedMemConfig = hip.hip.hipCtxGetSharedMemConfig
cuCtxSynchronize = hip.hip.hipCtxSynchronize
cuCtxGetFlags = hip.hip.hipCtxGetFlags
cuCtxEnablePeerAccess = hip.hip.hipCtxEnablePeerAccess
cuCtxDisablePeerAccess = hip.hip.hipCtxDisablePeerAccess
cuDevicePrimaryCtxGetState = hip.hip.hipDevicePrimaryCtxGetState
cuDevicePrimaryCtxRelease = hip.hip.hipDevicePrimaryCtxRelease
cuDevicePrimaryCtxRelease_v2 = hip.hip.hipDevicePrimaryCtxRelease
cuDevicePrimaryCtxRetain = hip.hip.hipDevicePrimaryCtxRetain
cuDevicePrimaryCtxReset = hip.hip.hipDevicePrimaryCtxReset
cuDevicePrimaryCtxReset_v2 = hip.hip.hipDevicePrimaryCtxReset
cuDevicePrimaryCtxSetFlags = hip.hip.hipDevicePrimaryCtxSetFlags
cuDevicePrimaryCtxSetFlags_v2 = hip.hip.hipDevicePrimaryCtxSetFlags
cuModuleLoad = hip.hip.hipModuleLoad
cuModuleUnload = hip.hip.hipModuleUnload
cuModuleGetFunction = hip.hip.hipModuleGetFunction
cudaFuncGetAttributes = hip.hip.hipFuncGetAttributes
cuFuncGetAttribute = hip.hip.hipFuncGetAttribute
cuModuleGetTexRef = hip.hip.hipModuleGetTexRef
cuModuleLoadData = hip.hip.hipModuleLoadData
cuModuleLoadDataEx = hip.hip.hipModuleLoadDataEx
cuLaunchKernel = hip.hip.hipModuleLaunchKernel
cudaLaunchCooperativeKernel = hip.hip.hipLaunchCooperativeKernel
cudaLaunchCooperativeKernelMultiDevice = hip.hip.hipLaunchCooperativeKernelMultiDevice
cuOccupancyMaxPotentialBlockSize = hip.hip.hipModuleOccupancyMaxPotentialBlockSize
cuOccupancyMaxPotentialBlockSizeWithFlags = hip.hip.hipModuleOccupancyMaxPotentialBlockSizeWithFlags
cuOccupancyMaxActiveBlocksPerMultiprocessor = hip.hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = hip.hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
cudaOccupancyMaxActiveBlocksPerMultiprocessor = hip.hip.hipOccupancyMaxActiveBlocksPerMultiprocessor
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = hip.hip.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
cudaOccupancyMaxPotentialBlockSize = hip.hip.hipOccupancyMaxPotentialBlockSize
cuProfilerStart = hip.hip.hipProfilerStart
cudaProfilerStart = hip.hip.hipProfilerStart
cuProfilerStop = hip.hip.hipProfilerStop
cudaProfilerStop = hip.hip.hipProfilerStop
cudaConfigureCall = hip.hip.hipConfigureCall
cudaSetupArgument = hip.hip.hipSetupArgument
cudaLaunch = hip.hip.hipLaunchByPtr
cudaLaunchKernel = hip.hip.hipLaunchKernel
cuLaunchHostFunc = hip.hip.hipLaunchHostFunc
cudaLaunchHostFunc = hip.hip.hipLaunchHostFunc
cuMemcpy2DUnaligned = hip.hip.hipDrvMemcpy2DUnaligned
cuMemcpy2DUnaligned_v2 = hip.hip.hipDrvMemcpy2DUnaligned
cudaBindTextureToMipmappedArray = hip.hip.hipBindTextureToMipmappedArray
cudaCreateTextureObject = hip.hip.hipCreateTextureObject
cudaDestroyTextureObject = hip.hip.hipDestroyTextureObject
cudaGetChannelDesc = hip.hip.hipGetChannelDesc
cudaGetTextureObjectResourceDesc = hip.hip.hipGetTextureObjectResourceDesc
cudaGetTextureObjectResourceViewDesc = hip.hip.hipGetTextureObjectResourceViewDesc
cudaGetTextureObjectTextureDesc = hip.hip.hipGetTextureObjectTextureDesc
cuTexObjectCreate = hip.hip.hipTexObjectCreate
cuTexObjectDestroy = hip.hip.hipTexObjectDestroy
cuTexObjectGetResourceDesc = hip.hip.hipTexObjectGetResourceDesc
cuTexObjectGetResourceViewDesc = hip.hip.hipTexObjectGetResourceViewDesc
cuTexObjectGetTextureDesc = hip.hip.hipTexObjectGetTextureDesc
cudaGetTextureReference = hip.hip.hipGetTextureReference
cuTexRefSetAddressMode = hip.hip.hipTexRefSetAddressMode
cuTexRefSetArray = hip.hip.hipTexRefSetArray
cuTexRefSetFilterMode = hip.hip.hipTexRefSetFilterMode
cuTexRefSetFlags = hip.hip.hipTexRefSetFlags
cuTexRefSetFormat = hip.hip.hipTexRefSetFormat
cudaBindTexture = hip.hip.hipBindTexture
cudaBindTexture2D = hip.hip.hipBindTexture2D
cudaBindTextureToArray = hip.hip.hipBindTextureToArray
cudaGetTextureAlignmentOffset = hip.hip.hipGetTextureAlignmentOffset
cudaUnbindTexture = hip.hip.hipUnbindTexture
cuTexRefGetAddress = hip.hip.hipTexRefGetAddress
cuTexRefGetAddress_v2 = hip.hip.hipTexRefGetAddress
cuTexRefGetAddressMode = hip.hip.hipTexRefGetAddressMode
cuTexRefGetFilterMode = hip.hip.hipTexRefGetFilterMode
cuTexRefGetFlags = hip.hip.hipTexRefGetFlags
cuTexRefGetFormat = hip.hip.hipTexRefGetFormat
cuTexRefGetMaxAnisotropy = hip.hip.hipTexRefGetMaxAnisotropy
cuTexRefGetMipmapFilterMode = hip.hip.hipTexRefGetMipmapFilterMode
cuTexRefGetMipmapLevelBias = hip.hip.hipTexRefGetMipmapLevelBias
cuTexRefGetMipmapLevelClamp = hip.hip.hipTexRefGetMipmapLevelClamp
cuTexRefGetMipmappedArray = hip.hip.hipTexRefGetMipMappedArray
cuTexRefSetAddress = hip.hip.hipTexRefSetAddress
cuTexRefSetAddress_v2 = hip.hip.hipTexRefSetAddress
cuTexRefSetAddress2D = hip.hip.hipTexRefSetAddress2D
cuTexRefSetAddress2D_v2 = hip.hip.hipTexRefSetAddress2D
cuTexRefSetAddress2D_v3 = hip.hip.hipTexRefSetAddress2D
cuTexRefSetMaxAnisotropy = hip.hip.hipTexRefSetMaxAnisotropy
cuTexRefSetBorderColor = hip.hip.hipTexRefSetBorderColor
cuTexRefSetMipmapFilterMode = hip.hip.hipTexRefSetMipmapFilterMode
cuTexRefSetMipmapLevelBias = hip.hip.hipTexRefSetMipmapLevelBias
cuTexRefSetMipmapLevelClamp = hip.hip.hipTexRefSetMipmapLevelClamp
cuTexRefSetMipmappedArray = hip.hip.hipTexRefSetMipmappedArray
cuMipmappedArrayCreate = hip.hip.hipMipmappedArrayCreate
cuMipmappedArrayDestroy = hip.hip.hipMipmappedArrayDestroy
cuMipmappedArrayGetLevel = hip.hip.hipMipmappedArrayGetLevel
cuStreamBeginCapture = hip.hip.hipStreamBeginCapture
cuStreamBeginCapture_v2 = hip.hip.hipStreamBeginCapture
cudaStreamBeginCapture = hip.hip.hipStreamBeginCapture
cuStreamEndCapture = hip.hip.hipStreamEndCapture
cudaStreamEndCapture = hip.hip.hipStreamEndCapture
cuStreamGetCaptureInfo = hip.hip.hipStreamGetCaptureInfo
cudaStreamGetCaptureInfo = hip.hip.hipStreamGetCaptureInfo
cuStreamGetCaptureInfo_v2 = hip.hip.hipStreamGetCaptureInfo_v2
cuStreamIsCapturing = hip.hip.hipStreamIsCapturing
cudaStreamIsCapturing = hip.hip.hipStreamIsCapturing
cuStreamUpdateCaptureDependencies = hip.hip.hipStreamUpdateCaptureDependencies
cuThreadExchangeStreamCaptureMode = hip.hip.hipThreadExchangeStreamCaptureMode
cudaThreadExchangeStreamCaptureMode = hip.hip.hipThreadExchangeStreamCaptureMode
cuGraphCreate = hip.hip.hipGraphCreate
cudaGraphCreate = hip.hip.hipGraphCreate
cuGraphDestroy = hip.hip.hipGraphDestroy
cudaGraphDestroy = hip.hip.hipGraphDestroy
cuGraphAddDependencies = hip.hip.hipGraphAddDependencies
cudaGraphAddDependencies = hip.hip.hipGraphAddDependencies
cuGraphRemoveDependencies = hip.hip.hipGraphRemoveDependencies
cudaGraphRemoveDependencies = hip.hip.hipGraphRemoveDependencies
cuGraphGetEdges = hip.hip.hipGraphGetEdges
cudaGraphGetEdges = hip.hip.hipGraphGetEdges
cuGraphGetNodes = hip.hip.hipGraphGetNodes
cudaGraphGetNodes = hip.hip.hipGraphGetNodes
cuGraphGetRootNodes = hip.hip.hipGraphGetRootNodes
cudaGraphGetRootNodes = hip.hip.hipGraphGetRootNodes
cuGraphNodeGetDependencies = hip.hip.hipGraphNodeGetDependencies
cudaGraphNodeGetDependencies = hip.hip.hipGraphNodeGetDependencies
cuGraphNodeGetDependentNodes = hip.hip.hipGraphNodeGetDependentNodes
cudaGraphNodeGetDependentNodes = hip.hip.hipGraphNodeGetDependentNodes
cuGraphNodeGetType = hip.hip.hipGraphNodeGetType
cudaGraphNodeGetType = hip.hip.hipGraphNodeGetType
cuGraphDestroyNode = hip.hip.hipGraphDestroyNode
cudaGraphDestroyNode = hip.hip.hipGraphDestroyNode
cuGraphClone = hip.hip.hipGraphClone
cudaGraphClone = hip.hip.hipGraphClone
cuGraphNodeFindInClone = hip.hip.hipGraphNodeFindInClone
cudaGraphNodeFindInClone = hip.hip.hipGraphNodeFindInClone
cuGraphInstantiate = hip.hip.hipGraphInstantiate
cuGraphInstantiate_v2 = hip.hip.hipGraphInstantiate
cudaGraphInstantiate = hip.hip.hipGraphInstantiate
cuGraphInstantiateWithFlags = hip.hip.hipGraphInstantiateWithFlags
cudaGraphInstantiateWithFlags = hip.hip.hipGraphInstantiateWithFlags
cuGraphLaunch = hip.hip.hipGraphLaunch
cudaGraphLaunch = hip.hip.hipGraphLaunch
cuGraphUpload = hip.hip.hipGraphUpload
cudaGraphUpload = hip.hip.hipGraphUpload
cuGraphExecDestroy = hip.hip.hipGraphExecDestroy
cudaGraphExecDestroy = hip.hip.hipGraphExecDestroy
cuGraphExecUpdate = hip.hip.hipGraphExecUpdate
cudaGraphExecUpdate = hip.hip.hipGraphExecUpdate
cuGraphAddKernelNode = hip.hip.hipGraphAddKernelNode
cudaGraphAddKernelNode = hip.hip.hipGraphAddKernelNode
cuGraphKernelNodeGetParams = hip.hip.hipGraphKernelNodeGetParams
cudaGraphKernelNodeGetParams = hip.hip.hipGraphKernelNodeGetParams
cuGraphKernelNodeSetParams = hip.hip.hipGraphKernelNodeSetParams
cudaGraphKernelNodeSetParams = hip.hip.hipGraphKernelNodeSetParams
cuGraphExecKernelNodeSetParams = hip.hip.hipGraphExecKernelNodeSetParams
cudaGraphExecKernelNodeSetParams = hip.hip.hipGraphExecKernelNodeSetParams
cudaGraphAddMemcpyNode = hip.hip.hipGraphAddMemcpyNode
cuGraphMemcpyNodeGetParams = hip.hip.hipGraphMemcpyNodeGetParams
cudaGraphMemcpyNodeGetParams = hip.hip.hipGraphMemcpyNodeGetParams
cuGraphMemcpyNodeSetParams = hip.hip.hipGraphMemcpyNodeSetParams
cudaGraphMemcpyNodeSetParams = hip.hip.hipGraphMemcpyNodeSetParams
cuGraphKernelNodeSetAttribute = hip.hip.hipGraphKernelNodeSetAttribute
cudaGraphKernelNodeSetAttribute = hip.hip.hipGraphKernelNodeSetAttribute
cuGraphKernelNodeGetAttribute = hip.hip.hipGraphKernelNodeGetAttribute
cudaGraphKernelNodeGetAttribute = hip.hip.hipGraphKernelNodeGetAttribute
cudaGraphExecMemcpyNodeSetParams = hip.hip.hipGraphExecMemcpyNodeSetParams
cudaGraphAddMemcpyNode1D = hip.hip.hipGraphAddMemcpyNode1D
cudaGraphMemcpyNodeSetParams1D = hip.hip.hipGraphMemcpyNodeSetParams1D
cudaGraphExecMemcpyNodeSetParams1D = hip.hip.hipGraphExecMemcpyNodeSetParams1D
cudaGraphAddMemcpyNodeFromSymbol = hip.hip.hipGraphAddMemcpyNodeFromSymbol
cudaGraphMemcpyNodeSetParamsFromSymbol = hip.hip.hipGraphMemcpyNodeSetParamsFromSymbol
cudaGraphExecMemcpyNodeSetParamsFromSymbol = hip.hip.hipGraphExecMemcpyNodeSetParamsFromSymbol
cudaGraphAddMemcpyNodeToSymbol = hip.hip.hipGraphAddMemcpyNodeToSymbol
cudaGraphMemcpyNodeSetParamsToSymbol = hip.hip.hipGraphMemcpyNodeSetParamsToSymbol
cudaGraphExecMemcpyNodeSetParamsToSymbol = hip.hip.hipGraphExecMemcpyNodeSetParamsToSymbol
cudaGraphAddMemsetNode = hip.hip.hipGraphAddMemsetNode
cuGraphMemsetNodeGetParams = hip.hip.hipGraphMemsetNodeGetParams
cudaGraphMemsetNodeGetParams = hip.hip.hipGraphMemsetNodeGetParams
cuGraphMemsetNodeSetParams = hip.hip.hipGraphMemsetNodeSetParams
cudaGraphMemsetNodeSetParams = hip.hip.hipGraphMemsetNodeSetParams
cudaGraphExecMemsetNodeSetParams = hip.hip.hipGraphExecMemsetNodeSetParams
cuGraphAddHostNode = hip.hip.hipGraphAddHostNode
cudaGraphAddHostNode = hip.hip.hipGraphAddHostNode
cuGraphHostNodeGetParams = hip.hip.hipGraphHostNodeGetParams
cudaGraphHostNodeGetParams = hip.hip.hipGraphHostNodeGetParams
cuGraphHostNodeSetParams = hip.hip.hipGraphHostNodeSetParams
cudaGraphHostNodeSetParams = hip.hip.hipGraphHostNodeSetParams
cuGraphExecHostNodeSetParams = hip.hip.hipGraphExecHostNodeSetParams
cudaGraphExecHostNodeSetParams = hip.hip.hipGraphExecHostNodeSetParams
cuGraphAddChildGraphNode = hip.hip.hipGraphAddChildGraphNode
cudaGraphAddChildGraphNode = hip.hip.hipGraphAddChildGraphNode
cuGraphChildGraphNodeGetGraph = hip.hip.hipGraphChildGraphNodeGetGraph
cudaGraphChildGraphNodeGetGraph = hip.hip.hipGraphChildGraphNodeGetGraph
cuGraphExecChildGraphNodeSetParams = hip.hip.hipGraphExecChildGraphNodeSetParams
cudaGraphExecChildGraphNodeSetParams = hip.hip.hipGraphExecChildGraphNodeSetParams
cuGraphAddEmptyNode = hip.hip.hipGraphAddEmptyNode
cudaGraphAddEmptyNode = hip.hip.hipGraphAddEmptyNode
cuGraphAddEventRecordNode = hip.hip.hipGraphAddEventRecordNode
cudaGraphAddEventRecordNode = hip.hip.hipGraphAddEventRecordNode
cuGraphEventRecordNodeGetEvent = hip.hip.hipGraphEventRecordNodeGetEvent
cudaGraphEventRecordNodeGetEvent = hip.hip.hipGraphEventRecordNodeGetEvent
cuGraphEventRecordNodeSetEvent = hip.hip.hipGraphEventRecordNodeSetEvent
cudaGraphEventRecordNodeSetEvent = hip.hip.hipGraphEventRecordNodeSetEvent
cuGraphExecEventRecordNodeSetEvent = hip.hip.hipGraphExecEventRecordNodeSetEvent
cudaGraphExecEventRecordNodeSetEvent = hip.hip.hipGraphExecEventRecordNodeSetEvent
cuGraphAddEventWaitNode = hip.hip.hipGraphAddEventWaitNode
cudaGraphAddEventWaitNode = hip.hip.hipGraphAddEventWaitNode
cuGraphEventWaitNodeGetEvent = hip.hip.hipGraphEventWaitNodeGetEvent
cudaGraphEventWaitNodeGetEvent = hip.hip.hipGraphEventWaitNodeGetEvent
cuGraphEventWaitNodeSetEvent = hip.hip.hipGraphEventWaitNodeSetEvent
cudaGraphEventWaitNodeSetEvent = hip.hip.hipGraphEventWaitNodeSetEvent
cuGraphExecEventWaitNodeSetEvent = hip.hip.hipGraphExecEventWaitNodeSetEvent
cudaGraphExecEventWaitNodeSetEvent = hip.hip.hipGraphExecEventWaitNodeSetEvent
cuDeviceGetGraphMemAttribute = hip.hip.hipDeviceGetGraphMemAttribute
cudaDeviceGetGraphMemAttribute = hip.hip.hipDeviceGetGraphMemAttribute
cuDeviceSetGraphMemAttribute = hip.hip.hipDeviceSetGraphMemAttribute
cudaDeviceSetGraphMemAttribute = hip.hip.hipDeviceSetGraphMemAttribute
cuDeviceGraphMemTrim = hip.hip.hipDeviceGraphMemTrim
cudaDeviceGraphMemTrim = hip.hip.hipDeviceGraphMemTrim
cuUserObjectCreate = hip.hip.hipUserObjectCreate
cudaUserObjectCreate = hip.hip.hipUserObjectCreate
cuUserObjectRelease = hip.hip.hipUserObjectRelease
cudaUserObjectRelease = hip.hip.hipUserObjectRelease
cuUserObjectRetain = hip.hip.hipUserObjectRetain
cudaUserObjectRetain = hip.hip.hipUserObjectRetain
cuGraphRetainUserObject = hip.hip.hipGraphRetainUserObject
cudaGraphRetainUserObject = hip.hip.hipGraphRetainUserObject
cuGraphReleaseUserObject = hip.hip.hipGraphReleaseUserObject
cudaGraphReleaseUserObject = hip.hip.hipGraphReleaseUserObject
cuMemAddressFree = hip.hip.hipMemAddressFree
cuMemAddressReserve = hip.hip.hipMemAddressReserve
cuMemCreate = hip.hip.hipMemCreate
cuMemExportToShareableHandle = hip.hip.hipMemExportToShareableHandle
cuMemGetAccess = hip.hip.hipMemGetAccess
cuMemGetAllocationGranularity = hip.hip.hipMemGetAllocationGranularity
cuMemGetAllocationPropertiesFromHandle = hip.hip.hipMemGetAllocationPropertiesFromHandle
cuMemImportFromShareableHandle = hip.hip.hipMemImportFromShareableHandle
cuMemMap = hip.hip.hipMemMap
cuMemMapArrayAsync = hip.hip.hipMemMapArrayAsync
cuMemRelease = hip.hip.hipMemRelease
cuMemRetainAllocationHandle = hip.hip.hipMemRetainAllocationHandle
cuMemSetAccess = hip.hip.hipMemSetAccess
cuMemUnmap = hip.hip.hipMemUnmap
cuGLGetDevices = hip.hip.hipGLGetDevices
cudaGLGetDevices = hip.hip.hipGLGetDevices
cuGraphicsGLRegisterBuffer = hip.hip.hipGraphicsGLRegisterBuffer
cudaGraphicsGLRegisterBuffer = hip.hip.hipGraphicsGLRegisterBuffer
cuGraphicsGLRegisterImage = hip.hip.hipGraphicsGLRegisterImage
cudaGraphicsGLRegisterImage = hip.hip.hipGraphicsGLRegisterImage
cuGraphicsMapResources = hip.hip.hipGraphicsMapResources
cudaGraphicsMapResources = hip.hip.hipGraphicsMapResources
cuGraphicsSubResourceGetMappedArray = hip.hip.hipGraphicsSubResourceGetMappedArray
cudaGraphicsSubResourceGetMappedArray = hip.hip.hipGraphicsSubResourceGetMappedArray
cuGraphicsResourceGetMappedPointer = hip.hip.hipGraphicsResourceGetMappedPointer
cuGraphicsResourceGetMappedPointer_v2 = hip.hip.hipGraphicsResourceGetMappedPointer
cudaGraphicsResourceGetMappedPointer = hip.hip.hipGraphicsResourceGetMappedPointer
cuGraphicsUnmapResources = hip.hip.hipGraphicsUnmapResources
cudaGraphicsUnmapResources = hip.hip.hipGraphicsUnmapResources
cuGraphicsUnregisterResource = hip.hip.hipGraphicsUnregisterResource
cudaGraphicsUnregisterResource = hip.hip.hipGraphicsUnregisterResource