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
    def get_bytes(self, i):
        """Get value of ``bytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].bytes
    @property
    def bytes(self):
        return self.get_bytes(0)
    # TODO is_basic_type_constantarray: add setters



cdef class hipUUID:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipUUID from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipUUID`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipUUID wrapper = hipUUID.__new__(hipUUID)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_name(self, i):
        """Get value of ``name`` of ``self._ptr[i]``.
        """
        return self._ptr[i].name
    @property
    def name(self):
        return self.get_name(0)
    # TODO is_basic_type_constantarray: add setters
    def get_totalGlobalMem(self, i):
        """Get value ``totalGlobalMem`` of ``self._ptr[i]``.
        """
        return self._ptr[i].totalGlobalMem
    def set_totalGlobalMem(self, i, int value):
        """Set value ``totalGlobalMem`` of ``self._ptr[i]``.
        """
        self._ptr[i].totalGlobalMem = value
    @property
    def totalGlobalMem(self):
        return self.get_totalGlobalMem(0)
    @totalGlobalMem.setter
    def totalGlobalMem(self, int value):
        self.set_totalGlobalMem(0,value)
    def get_sharedMemPerBlock(self, i):
        """Get value ``sharedMemPerBlock`` of ``self._ptr[i]``.
        """
        return self._ptr[i].sharedMemPerBlock
    def set_sharedMemPerBlock(self, i, int value):
        """Set value ``sharedMemPerBlock`` of ``self._ptr[i]``.
        """
        self._ptr[i].sharedMemPerBlock = value
    @property
    def sharedMemPerBlock(self):
        return self.get_sharedMemPerBlock(0)
    @sharedMemPerBlock.setter
    def sharedMemPerBlock(self, int value):
        self.set_sharedMemPerBlock(0,value)
    def get_regsPerBlock(self, i):
        """Get value ``regsPerBlock`` of ``self._ptr[i]``.
        """
        return self._ptr[i].regsPerBlock
    def set_regsPerBlock(self, i, int value):
        """Set value ``regsPerBlock`` of ``self._ptr[i]``.
        """
        self._ptr[i].regsPerBlock = value
    @property
    def regsPerBlock(self):
        return self.get_regsPerBlock(0)
    @regsPerBlock.setter
    def regsPerBlock(self, int value):
        self.set_regsPerBlock(0,value)
    def get_warpSize(self, i):
        """Get value ``warpSize`` of ``self._ptr[i]``.
        """
        return self._ptr[i].warpSize
    def set_warpSize(self, i, int value):
        """Set value ``warpSize`` of ``self._ptr[i]``.
        """
        self._ptr[i].warpSize = value
    @property
    def warpSize(self):
        return self.get_warpSize(0)
    @warpSize.setter
    def warpSize(self, int value):
        self.set_warpSize(0,value)
    def get_maxThreadsPerBlock(self, i):
        """Get value ``maxThreadsPerBlock`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxThreadsPerBlock
    def set_maxThreadsPerBlock(self, i, int value):
        """Set value ``maxThreadsPerBlock`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxThreadsPerBlock = value
    @property
    def maxThreadsPerBlock(self):
        return self.get_maxThreadsPerBlock(0)
    @maxThreadsPerBlock.setter
    def maxThreadsPerBlock(self, int value):
        self.set_maxThreadsPerBlock(0,value)
    def get_maxThreadsDim(self, i):
        """Get value of ``maxThreadsDim`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxThreadsDim
    @property
    def maxThreadsDim(self):
        return self.get_maxThreadsDim(0)
    # TODO is_basic_type_constantarray: add setters
    def get_maxGridSize(self, i):
        """Get value of ``maxGridSize`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxGridSize
    @property
    def maxGridSize(self):
        return self.get_maxGridSize(0)
    # TODO is_basic_type_constantarray: add setters
    def get_clockRate(self, i):
        """Get value ``clockRate`` of ``self._ptr[i]``.
        """
        return self._ptr[i].clockRate
    def set_clockRate(self, i, int value):
        """Set value ``clockRate`` of ``self._ptr[i]``.
        """
        self._ptr[i].clockRate = value
    @property
    def clockRate(self):
        return self.get_clockRate(0)
    @clockRate.setter
    def clockRate(self, int value):
        self.set_clockRate(0,value)
    def get_memoryClockRate(self, i):
        """Get value ``memoryClockRate`` of ``self._ptr[i]``.
        """
        return self._ptr[i].memoryClockRate
    def set_memoryClockRate(self, i, int value):
        """Set value ``memoryClockRate`` of ``self._ptr[i]``.
        """
        self._ptr[i].memoryClockRate = value
    @property
    def memoryClockRate(self):
        return self.get_memoryClockRate(0)
    @memoryClockRate.setter
    def memoryClockRate(self, int value):
        self.set_memoryClockRate(0,value)
    def get_memoryBusWidth(self, i):
        """Get value ``memoryBusWidth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].memoryBusWidth
    def set_memoryBusWidth(self, i, int value):
        """Set value ``memoryBusWidth`` of ``self._ptr[i]``.
        """
        self._ptr[i].memoryBusWidth = value
    @property
    def memoryBusWidth(self):
        return self.get_memoryBusWidth(0)
    @memoryBusWidth.setter
    def memoryBusWidth(self, int value):
        self.set_memoryBusWidth(0,value)
    def get_totalConstMem(self, i):
        """Get value ``totalConstMem`` of ``self._ptr[i]``.
        """
        return self._ptr[i].totalConstMem
    def set_totalConstMem(self, i, int value):
        """Set value ``totalConstMem`` of ``self._ptr[i]``.
        """
        self._ptr[i].totalConstMem = value
    @property
    def totalConstMem(self):
        return self.get_totalConstMem(0)
    @totalConstMem.setter
    def totalConstMem(self, int value):
        self.set_totalConstMem(0,value)
    def get_major(self, i):
        """Get value ``major`` of ``self._ptr[i]``.
        """
        return self._ptr[i].major
    def set_major(self, i, int value):
        """Set value ``major`` of ``self._ptr[i]``.
        """
        self._ptr[i].major = value
    @property
    def major(self):
        return self.get_major(0)
    @major.setter
    def major(self, int value):
        self.set_major(0,value)
    def get_minor(self, i):
        """Get value ``minor`` of ``self._ptr[i]``.
        """
        return self._ptr[i].minor
    def set_minor(self, i, int value):
        """Set value ``minor`` of ``self._ptr[i]``.
        """
        self._ptr[i].minor = value
    @property
    def minor(self):
        return self.get_minor(0)
    @minor.setter
    def minor(self, int value):
        self.set_minor(0,value)
    def get_multiProcessorCount(self, i):
        """Get value ``multiProcessorCount`` of ``self._ptr[i]``.
        """
        return self._ptr[i].multiProcessorCount
    def set_multiProcessorCount(self, i, int value):
        """Set value ``multiProcessorCount`` of ``self._ptr[i]``.
        """
        self._ptr[i].multiProcessorCount = value
    @property
    def multiProcessorCount(self):
        return self.get_multiProcessorCount(0)
    @multiProcessorCount.setter
    def multiProcessorCount(self, int value):
        self.set_multiProcessorCount(0,value)
    def get_l2CacheSize(self, i):
        """Get value ``l2CacheSize`` of ``self._ptr[i]``.
        """
        return self._ptr[i].l2CacheSize
    def set_l2CacheSize(self, i, int value):
        """Set value ``l2CacheSize`` of ``self._ptr[i]``.
        """
        self._ptr[i].l2CacheSize = value
    @property
    def l2CacheSize(self):
        return self.get_l2CacheSize(0)
    @l2CacheSize.setter
    def l2CacheSize(self, int value):
        self.set_l2CacheSize(0,value)
    def get_maxThreadsPerMultiProcessor(self, i):
        """Get value ``maxThreadsPerMultiProcessor`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxThreadsPerMultiProcessor
    def set_maxThreadsPerMultiProcessor(self, i, int value):
        """Set value ``maxThreadsPerMultiProcessor`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxThreadsPerMultiProcessor = value
    @property
    def maxThreadsPerMultiProcessor(self):
        return self.get_maxThreadsPerMultiProcessor(0)
    @maxThreadsPerMultiProcessor.setter
    def maxThreadsPerMultiProcessor(self, int value):
        self.set_maxThreadsPerMultiProcessor(0,value)
    def get_computeMode(self, i):
        """Get value ``computeMode`` of ``self._ptr[i]``.
        """
        return self._ptr[i].computeMode
    def set_computeMode(self, i, int value):
        """Set value ``computeMode`` of ``self._ptr[i]``.
        """
        self._ptr[i].computeMode = value
    @property
    def computeMode(self):
        return self.get_computeMode(0)
    @computeMode.setter
    def computeMode(self, int value):
        self.set_computeMode(0,value)
    def get_clockInstructionRate(self, i):
        """Get value ``clockInstructionRate`` of ``self._ptr[i]``.
        """
        return self._ptr[i].clockInstructionRate
    def set_clockInstructionRate(self, i, int value):
        """Set value ``clockInstructionRate`` of ``self._ptr[i]``.
        """
        self._ptr[i].clockInstructionRate = value
    @property
    def clockInstructionRate(self):
        return self.get_clockInstructionRate(0)
    @clockInstructionRate.setter
    def clockInstructionRate(self, int value):
        self.set_clockInstructionRate(0,value)
    def get_arch(self, i):
        """Get value of ``arch`` of ``self._ptr[i]``.
        """
        return hipDeviceArch_t.from_ptr(&self._ptr[i].arch)
    @property
    def arch(self):
        return self.get_arch(0)
    def get_concurrentKernels(self, i):
        """Get value ``concurrentKernels`` of ``self._ptr[i]``.
        """
        return self._ptr[i].concurrentKernels
    def set_concurrentKernels(self, i, int value):
        """Set value ``concurrentKernels`` of ``self._ptr[i]``.
        """
        self._ptr[i].concurrentKernels = value
    @property
    def concurrentKernels(self):
        return self.get_concurrentKernels(0)
    @concurrentKernels.setter
    def concurrentKernels(self, int value):
        self.set_concurrentKernels(0,value)
    def get_pciDomainID(self, i):
        """Get value ``pciDomainID`` of ``self._ptr[i]``.
        """
        return self._ptr[i].pciDomainID
    def set_pciDomainID(self, i, int value):
        """Set value ``pciDomainID`` of ``self._ptr[i]``.
        """
        self._ptr[i].pciDomainID = value
    @property
    def pciDomainID(self):
        return self.get_pciDomainID(0)
    @pciDomainID.setter
    def pciDomainID(self, int value):
        self.set_pciDomainID(0,value)
    def get_pciBusID(self, i):
        """Get value ``pciBusID`` of ``self._ptr[i]``.
        """
        return self._ptr[i].pciBusID
    def set_pciBusID(self, i, int value):
        """Set value ``pciBusID`` of ``self._ptr[i]``.
        """
        self._ptr[i].pciBusID = value
    @property
    def pciBusID(self):
        return self.get_pciBusID(0)
    @pciBusID.setter
    def pciBusID(self, int value):
        self.set_pciBusID(0,value)
    def get_pciDeviceID(self, i):
        """Get value ``pciDeviceID`` of ``self._ptr[i]``.
        """
        return self._ptr[i].pciDeviceID
    def set_pciDeviceID(self, i, int value):
        """Set value ``pciDeviceID`` of ``self._ptr[i]``.
        """
        self._ptr[i].pciDeviceID = value
    @property
    def pciDeviceID(self):
        return self.get_pciDeviceID(0)
    @pciDeviceID.setter
    def pciDeviceID(self, int value):
        self.set_pciDeviceID(0,value)
    def get_maxSharedMemoryPerMultiProcessor(self, i):
        """Get value ``maxSharedMemoryPerMultiProcessor`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxSharedMemoryPerMultiProcessor
    def set_maxSharedMemoryPerMultiProcessor(self, i, int value):
        """Set value ``maxSharedMemoryPerMultiProcessor`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxSharedMemoryPerMultiProcessor = value
    @property
    def maxSharedMemoryPerMultiProcessor(self):
        return self.get_maxSharedMemoryPerMultiProcessor(0)
    @maxSharedMemoryPerMultiProcessor.setter
    def maxSharedMemoryPerMultiProcessor(self, int value):
        self.set_maxSharedMemoryPerMultiProcessor(0,value)
    def get_isMultiGpuBoard(self, i):
        """Get value ``isMultiGpuBoard`` of ``self._ptr[i]``.
        """
        return self._ptr[i].isMultiGpuBoard
    def set_isMultiGpuBoard(self, i, int value):
        """Set value ``isMultiGpuBoard`` of ``self._ptr[i]``.
        """
        self._ptr[i].isMultiGpuBoard = value
    @property
    def isMultiGpuBoard(self):
        return self.get_isMultiGpuBoard(0)
    @isMultiGpuBoard.setter
    def isMultiGpuBoard(self, int value):
        self.set_isMultiGpuBoard(0,value)
    def get_canMapHostMemory(self, i):
        """Get value ``canMapHostMemory`` of ``self._ptr[i]``.
        """
        return self._ptr[i].canMapHostMemory
    def set_canMapHostMemory(self, i, int value):
        """Set value ``canMapHostMemory`` of ``self._ptr[i]``.
        """
        self._ptr[i].canMapHostMemory = value
    @property
    def canMapHostMemory(self):
        return self.get_canMapHostMemory(0)
    @canMapHostMemory.setter
    def canMapHostMemory(self, int value):
        self.set_canMapHostMemory(0,value)
    def get_gcnArch(self, i):
        """Get value ``gcnArch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].gcnArch
    def set_gcnArch(self, i, int value):
        """Set value ``gcnArch`` of ``self._ptr[i]``.
        """
        self._ptr[i].gcnArch = value
    @property
    def gcnArch(self):
        return self.get_gcnArch(0)
    @gcnArch.setter
    def gcnArch(self, int value):
        self.set_gcnArch(0,value)
    def get_gcnArchName(self, i):
        """Get value of ``gcnArchName`` of ``self._ptr[i]``.
        """
        return self._ptr[i].gcnArchName
    @property
    def gcnArchName(self):
        return self.get_gcnArchName(0)
    # TODO is_basic_type_constantarray: add setters
    def get_integrated(self, i):
        """Get value ``integrated`` of ``self._ptr[i]``.
        """
        return self._ptr[i].integrated
    def set_integrated(self, i, int value):
        """Set value ``integrated`` of ``self._ptr[i]``.
        """
        self._ptr[i].integrated = value
    @property
    def integrated(self):
        return self.get_integrated(0)
    @integrated.setter
    def integrated(self, int value):
        self.set_integrated(0,value)
    def get_cooperativeLaunch(self, i):
        """Get value ``cooperativeLaunch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].cooperativeLaunch
    def set_cooperativeLaunch(self, i, int value):
        """Set value ``cooperativeLaunch`` of ``self._ptr[i]``.
        """
        self._ptr[i].cooperativeLaunch = value
    @property
    def cooperativeLaunch(self):
        return self.get_cooperativeLaunch(0)
    @cooperativeLaunch.setter
    def cooperativeLaunch(self, int value):
        self.set_cooperativeLaunch(0,value)
    def get_cooperativeMultiDeviceLaunch(self, i):
        """Get value ``cooperativeMultiDeviceLaunch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].cooperativeMultiDeviceLaunch
    def set_cooperativeMultiDeviceLaunch(self, i, int value):
        """Set value ``cooperativeMultiDeviceLaunch`` of ``self._ptr[i]``.
        """
        self._ptr[i].cooperativeMultiDeviceLaunch = value
    @property
    def cooperativeMultiDeviceLaunch(self):
        return self.get_cooperativeMultiDeviceLaunch(0)
    @cooperativeMultiDeviceLaunch.setter
    def cooperativeMultiDeviceLaunch(self, int value):
        self.set_cooperativeMultiDeviceLaunch(0,value)
    def get_maxTexture1DLinear(self, i):
        """Get value ``maxTexture1DLinear`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxTexture1DLinear
    def set_maxTexture1DLinear(self, i, int value):
        """Set value ``maxTexture1DLinear`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxTexture1DLinear = value
    @property
    def maxTexture1DLinear(self):
        return self.get_maxTexture1DLinear(0)
    @maxTexture1DLinear.setter
    def maxTexture1DLinear(self, int value):
        self.set_maxTexture1DLinear(0,value)
    def get_maxTexture1D(self, i):
        """Get value ``maxTexture1D`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxTexture1D
    def set_maxTexture1D(self, i, int value):
        """Set value ``maxTexture1D`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxTexture1D = value
    @property
    def maxTexture1D(self):
        return self.get_maxTexture1D(0)
    @maxTexture1D.setter
    def maxTexture1D(self, int value):
        self.set_maxTexture1D(0,value)
    def get_maxTexture2D(self, i):
        """Get value of ``maxTexture2D`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxTexture2D
    @property
    def maxTexture2D(self):
        return self.get_maxTexture2D(0)
    # TODO is_basic_type_constantarray: add setters
    def get_maxTexture3D(self, i):
        """Get value of ``maxTexture3D`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxTexture3D
    @property
    def maxTexture3D(self):
        return self.get_maxTexture3D(0)
    # TODO is_basic_type_constantarray: add setters
    def get_memPitch(self, i):
        """Get value ``memPitch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].memPitch
    def set_memPitch(self, i, int value):
        """Set value ``memPitch`` of ``self._ptr[i]``.
        """
        self._ptr[i].memPitch = value
    @property
    def memPitch(self):
        return self.get_memPitch(0)
    @memPitch.setter
    def memPitch(self, int value):
        self.set_memPitch(0,value)
    def get_textureAlignment(self, i):
        """Get value ``textureAlignment`` of ``self._ptr[i]``.
        """
        return self._ptr[i].textureAlignment
    def set_textureAlignment(self, i, int value):
        """Set value ``textureAlignment`` of ``self._ptr[i]``.
        """
        self._ptr[i].textureAlignment = value
    @property
    def textureAlignment(self):
        return self.get_textureAlignment(0)
    @textureAlignment.setter
    def textureAlignment(self, int value):
        self.set_textureAlignment(0,value)
    def get_texturePitchAlignment(self, i):
        """Get value ``texturePitchAlignment`` of ``self._ptr[i]``.
        """
        return self._ptr[i].texturePitchAlignment
    def set_texturePitchAlignment(self, i, int value):
        """Set value ``texturePitchAlignment`` of ``self._ptr[i]``.
        """
        self._ptr[i].texturePitchAlignment = value
    @property
    def texturePitchAlignment(self):
        return self.get_texturePitchAlignment(0)
    @texturePitchAlignment.setter
    def texturePitchAlignment(self, int value):
        self.set_texturePitchAlignment(0,value)
    def get_kernelExecTimeoutEnabled(self, i):
        """Get value ``kernelExecTimeoutEnabled`` of ``self._ptr[i]``.
        """
        return self._ptr[i].kernelExecTimeoutEnabled
    def set_kernelExecTimeoutEnabled(self, i, int value):
        """Set value ``kernelExecTimeoutEnabled`` of ``self._ptr[i]``.
        """
        self._ptr[i].kernelExecTimeoutEnabled = value
    @property
    def kernelExecTimeoutEnabled(self):
        return self.get_kernelExecTimeoutEnabled(0)
    @kernelExecTimeoutEnabled.setter
    def kernelExecTimeoutEnabled(self, int value):
        self.set_kernelExecTimeoutEnabled(0,value)
    def get_ECCEnabled(self, i):
        """Get value ``ECCEnabled`` of ``self._ptr[i]``.
        """
        return self._ptr[i].ECCEnabled
    def set_ECCEnabled(self, i, int value):
        """Set value ``ECCEnabled`` of ``self._ptr[i]``.
        """
        self._ptr[i].ECCEnabled = value
    @property
    def ECCEnabled(self):
        return self.get_ECCEnabled(0)
    @ECCEnabled.setter
    def ECCEnabled(self, int value):
        self.set_ECCEnabled(0,value)
    def get_tccDriver(self, i):
        """Get value ``tccDriver`` of ``self._ptr[i]``.
        """
        return self._ptr[i].tccDriver
    def set_tccDriver(self, i, int value):
        """Set value ``tccDriver`` of ``self._ptr[i]``.
        """
        self._ptr[i].tccDriver = value
    @property
    def tccDriver(self):
        return self.get_tccDriver(0)
    @tccDriver.setter
    def tccDriver(self, int value):
        self.set_tccDriver(0,value)
    def get_cooperativeMultiDeviceUnmatchedFunc(self, i):
        """Get value ``cooperativeMultiDeviceUnmatchedFunc`` of ``self._ptr[i]``.
        """
        return self._ptr[i].cooperativeMultiDeviceUnmatchedFunc
    def set_cooperativeMultiDeviceUnmatchedFunc(self, i, int value):
        """Set value ``cooperativeMultiDeviceUnmatchedFunc`` of ``self._ptr[i]``.
        """
        self._ptr[i].cooperativeMultiDeviceUnmatchedFunc = value
    @property
    def cooperativeMultiDeviceUnmatchedFunc(self):
        return self.get_cooperativeMultiDeviceUnmatchedFunc(0)
    @cooperativeMultiDeviceUnmatchedFunc.setter
    def cooperativeMultiDeviceUnmatchedFunc(self, int value):
        self.set_cooperativeMultiDeviceUnmatchedFunc(0,value)
    def get_cooperativeMultiDeviceUnmatchedGridDim(self, i):
        """Get value ``cooperativeMultiDeviceUnmatchedGridDim`` of ``self._ptr[i]``.
        """
        return self._ptr[i].cooperativeMultiDeviceUnmatchedGridDim
    def set_cooperativeMultiDeviceUnmatchedGridDim(self, i, int value):
        """Set value ``cooperativeMultiDeviceUnmatchedGridDim`` of ``self._ptr[i]``.
        """
        self._ptr[i].cooperativeMultiDeviceUnmatchedGridDim = value
    @property
    def cooperativeMultiDeviceUnmatchedGridDim(self):
        return self.get_cooperativeMultiDeviceUnmatchedGridDim(0)
    @cooperativeMultiDeviceUnmatchedGridDim.setter
    def cooperativeMultiDeviceUnmatchedGridDim(self, int value):
        self.set_cooperativeMultiDeviceUnmatchedGridDim(0,value)
    def get_cooperativeMultiDeviceUnmatchedBlockDim(self, i):
        """Get value ``cooperativeMultiDeviceUnmatchedBlockDim`` of ``self._ptr[i]``.
        """
        return self._ptr[i].cooperativeMultiDeviceUnmatchedBlockDim
    def set_cooperativeMultiDeviceUnmatchedBlockDim(self, i, int value):
        """Set value ``cooperativeMultiDeviceUnmatchedBlockDim`` of ``self._ptr[i]``.
        """
        self._ptr[i].cooperativeMultiDeviceUnmatchedBlockDim = value
    @property
    def cooperativeMultiDeviceUnmatchedBlockDim(self):
        return self.get_cooperativeMultiDeviceUnmatchedBlockDim(0)
    @cooperativeMultiDeviceUnmatchedBlockDim.setter
    def cooperativeMultiDeviceUnmatchedBlockDim(self, int value):
        self.set_cooperativeMultiDeviceUnmatchedBlockDim(0,value)
    def get_cooperativeMultiDeviceUnmatchedSharedMem(self, i):
        """Get value ``cooperativeMultiDeviceUnmatchedSharedMem`` of ``self._ptr[i]``.
        """
        return self._ptr[i].cooperativeMultiDeviceUnmatchedSharedMem
    def set_cooperativeMultiDeviceUnmatchedSharedMem(self, i, int value):
        """Set value ``cooperativeMultiDeviceUnmatchedSharedMem`` of ``self._ptr[i]``.
        """
        self._ptr[i].cooperativeMultiDeviceUnmatchedSharedMem = value
    @property
    def cooperativeMultiDeviceUnmatchedSharedMem(self):
        return self.get_cooperativeMultiDeviceUnmatchedSharedMem(0)
    @cooperativeMultiDeviceUnmatchedSharedMem.setter
    def cooperativeMultiDeviceUnmatchedSharedMem(self, int value):
        self.set_cooperativeMultiDeviceUnmatchedSharedMem(0,value)
    def get_isLargeBar(self, i):
        """Get value ``isLargeBar`` of ``self._ptr[i]``.
        """
        return self._ptr[i].isLargeBar
    def set_isLargeBar(self, i, int value):
        """Set value ``isLargeBar`` of ``self._ptr[i]``.
        """
        self._ptr[i].isLargeBar = value
    @property
    def isLargeBar(self):
        return self.get_isLargeBar(0)
    @isLargeBar.setter
    def isLargeBar(self, int value):
        self.set_isLargeBar(0,value)
    def get_asicRevision(self, i):
        """Get value ``asicRevision`` of ``self._ptr[i]``.
        """
        return self._ptr[i].asicRevision
    def set_asicRevision(self, i, int value):
        """Set value ``asicRevision`` of ``self._ptr[i]``.
        """
        self._ptr[i].asicRevision = value
    @property
    def asicRevision(self):
        return self.get_asicRevision(0)
    @asicRevision.setter
    def asicRevision(self, int value):
        self.set_asicRevision(0,value)
    def get_managedMemory(self, i):
        """Get value ``managedMemory`` of ``self._ptr[i]``.
        """
        return self._ptr[i].managedMemory
    def set_managedMemory(self, i, int value):
        """Set value ``managedMemory`` of ``self._ptr[i]``.
        """
        self._ptr[i].managedMemory = value
    @property
    def managedMemory(self):
        return self.get_managedMemory(0)
    @managedMemory.setter
    def managedMemory(self, int value):
        self.set_managedMemory(0,value)
    def get_directManagedMemAccessFromHost(self, i):
        """Get value ``directManagedMemAccessFromHost`` of ``self._ptr[i]``.
        """
        return self._ptr[i].directManagedMemAccessFromHost
    def set_directManagedMemAccessFromHost(self, i, int value):
        """Set value ``directManagedMemAccessFromHost`` of ``self._ptr[i]``.
        """
        self._ptr[i].directManagedMemAccessFromHost = value
    @property
    def directManagedMemAccessFromHost(self):
        return self.get_directManagedMemAccessFromHost(0)
    @directManagedMemAccessFromHost.setter
    def directManagedMemAccessFromHost(self, int value):
        self.set_directManagedMemAccessFromHost(0,value)
    def get_concurrentManagedAccess(self, i):
        """Get value ``concurrentManagedAccess`` of ``self._ptr[i]``.
        """
        return self._ptr[i].concurrentManagedAccess
    def set_concurrentManagedAccess(self, i, int value):
        """Set value ``concurrentManagedAccess`` of ``self._ptr[i]``.
        """
        self._ptr[i].concurrentManagedAccess = value
    @property
    def concurrentManagedAccess(self):
        return self.get_concurrentManagedAccess(0)
    @concurrentManagedAccess.setter
    def concurrentManagedAccess(self, int value):
        self.set_concurrentManagedAccess(0,value)
    def get_pageableMemoryAccess(self, i):
        """Get value ``pageableMemoryAccess`` of ``self._ptr[i]``.
        """
        return self._ptr[i].pageableMemoryAccess
    def set_pageableMemoryAccess(self, i, int value):
        """Set value ``pageableMemoryAccess`` of ``self._ptr[i]``.
        """
        self._ptr[i].pageableMemoryAccess = value
    @property
    def pageableMemoryAccess(self):
        return self.get_pageableMemoryAccess(0)
    @pageableMemoryAccess.setter
    def pageableMemoryAccess(self, int value):
        self.set_pageableMemoryAccess(0,value)
    def get_pageableMemoryAccessUsesHostPageTables(self, i):
        """Get value ``pageableMemoryAccessUsesHostPageTables`` of ``self._ptr[i]``.
        """
        return self._ptr[i].pageableMemoryAccessUsesHostPageTables
    def set_pageableMemoryAccessUsesHostPageTables(self, i, int value):
        """Set value ``pageableMemoryAccessUsesHostPageTables`` of ``self._ptr[i]``.
        """
        self._ptr[i].pageableMemoryAccessUsesHostPageTables = value
    @property
    def pageableMemoryAccessUsesHostPageTables(self):
        return self.get_pageableMemoryAccessUsesHostPageTables(0)
    @pageableMemoryAccessUsesHostPageTables.setter
    def pageableMemoryAccessUsesHostPageTables(self, int value):
        self.set_pageableMemoryAccessUsesHostPageTables(0,value)


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
    def get_memoryType(self, i):
        """Get value of ``memoryType`` of ``self._ptr[i]``.
        """
        return hipMemoryType(self._ptr[i].memoryType)
    def set_memoryType(self, i, value):
        """Set value ``memoryType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].memoryType = value.value
    @property
    def memoryType(self):
        return self.get_memoryType(0)
    @memoryType.setter
    def memoryType(self, value):
        self.set_memoryType(0,value)
    def get_device(self, i):
        """Get value ``device`` of ``self._ptr[i]``.
        """
        return self._ptr[i].device
    def set_device(self, i, int value):
        """Set value ``device`` of ``self._ptr[i]``.
        """
        self._ptr[i].device = value
    @property
    def device(self):
        return self.get_device(0)
    @device.setter
    def device(self, int value):
        self.set_device(0,value)
    def get_isManaged(self, i):
        """Get value ``isManaged`` of ``self._ptr[i]``.
        """
        return self._ptr[i].isManaged
    def set_isManaged(self, i, int value):
        """Set value ``isManaged`` of ``self._ptr[i]``.
        """
        self._ptr[i].isManaged = value
    @property
    def isManaged(self):
        return self.get_isManaged(0)
    @isManaged.setter
    def isManaged(self, int value):
        self.set_isManaged(0,value)
    def get_allocationFlags(self, i):
        """Get value ``allocationFlags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].allocationFlags
    def set_allocationFlags(self, i, unsigned int value):
        """Set value ``allocationFlags`` of ``self._ptr[i]``.
        """
        self._ptr[i].allocationFlags = value
    @property
    def allocationFlags(self):
        return self.get_allocationFlags(0)
    @allocationFlags.setter
    def allocationFlags(self, unsigned int value):
        self.set_allocationFlags(0,value)


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


cdef class hipDeviceptr_t:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipDeviceptr_t from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipDeviceptr_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipDeviceptr_t wrapper = hipDeviceptr_t.__new__(hipDeviceptr_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


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
    def get_x(self, i):
        """Get value ``x`` of ``self._ptr[i]``.
        """
        return self._ptr[i].x
    def set_x(self, i, int value):
        """Set value ``x`` of ``self._ptr[i]``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        return self.get_x(0)
    @x.setter
    def x(self, int value):
        self.set_x(0,value)
    def get_y(self, i):
        """Get value ``y`` of ``self._ptr[i]``.
        """
        return self._ptr[i].y
    def set_y(self, i, int value):
        """Set value ``y`` of ``self._ptr[i]``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        return self.get_y(0)
    @y.setter
    def y(self, int value):
        self.set_y(0,value)
    def get_z(self, i):
        """Get value ``z`` of ``self._ptr[i]``.
        """
        return self._ptr[i].z
    def set_z(self, i, int value):
        """Set value ``z`` of ``self._ptr[i]``.
        """
        self._ptr[i].z = value
    @property
    def z(self):
        return self.get_z(0)
    @z.setter
    def z(self, int value):
        self.set_z(0,value)
    def get_w(self, i):
        """Get value ``w`` of ``self._ptr[i]``.
        """
        return self._ptr[i].w
    def set_w(self, i, int value):
        """Set value ``w`` of ``self._ptr[i]``.
        """
        self._ptr[i].w = value
    @property
    def w(self):
        return self.get_w(0)
    @w.setter
    def w(self, int value):
        self.set_w(0,value)
    def get_f(self, i):
        """Get value of ``f`` of ``self._ptr[i]``.
        """
        return hipChannelFormatKind(self._ptr[i].f)
    def set_f(self, i, value):
        """Set value ``f`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipChannelFormatKind):
            raise TypeError("'value' must be of type 'hipChannelFormatKind'")
        self._ptr[i].f = value.value
    @property
    def f(self):
        return self.get_f(0)
    @f.setter
    def f(self, value):
        self.set_f(0,value)


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
    def get_Width(self, i):
        """Get value ``Width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].Width
    def set_Width(self, i, int value):
        """Set value ``Width`` of ``self._ptr[i]``.
        """
        self._ptr[i].Width = value
    @property
    def Width(self):
        return self.get_Width(0)
    @Width.setter
    def Width(self, int value):
        self.set_Width(0,value)
    def get_Height(self, i):
        """Get value ``Height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].Height
    def set_Height(self, i, int value):
        """Set value ``Height`` of ``self._ptr[i]``.
        """
        self._ptr[i].Height = value
    @property
    def Height(self):
        return self.get_Height(0)
    @Height.setter
    def Height(self, int value):
        self.set_Height(0,value)
    def get_Format(self, i):
        """Get value of ``Format`` of ``self._ptr[i]``.
        """
        return hipArray_Format(self._ptr[i].Format)
    def set_Format(self, i, value):
        """Set value ``Format`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].Format = value.value
    @property
    def Format(self):
        return self.get_Format(0)
    @Format.setter
    def Format(self, value):
        self.set_Format(0,value)
    def get_NumChannels(self, i):
        """Get value ``NumChannels`` of ``self._ptr[i]``.
        """
        return self._ptr[i].NumChannels
    def set_NumChannels(self, i, unsigned int value):
        """Set value ``NumChannels`` of ``self._ptr[i]``.
        """
        self._ptr[i].NumChannels = value
    @property
    def NumChannels(self):
        return self.get_NumChannels(0)
    @NumChannels.setter
    def NumChannels(self, unsigned int value):
        self.set_NumChannels(0,value)



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
    def get_Width(self, i):
        """Get value ``Width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].Width
    def set_Width(self, i, int value):
        """Set value ``Width`` of ``self._ptr[i]``.
        """
        self._ptr[i].Width = value
    @property
    def Width(self):
        return self.get_Width(0)
    @Width.setter
    def Width(self, int value):
        self.set_Width(0,value)
    def get_Height(self, i):
        """Get value ``Height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].Height
    def set_Height(self, i, int value):
        """Set value ``Height`` of ``self._ptr[i]``.
        """
        self._ptr[i].Height = value
    @property
    def Height(self):
        return self.get_Height(0)
    @Height.setter
    def Height(self, int value):
        self.set_Height(0,value)
    def get_Depth(self, i):
        """Get value ``Depth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].Depth
    def set_Depth(self, i, int value):
        """Set value ``Depth`` of ``self._ptr[i]``.
        """
        self._ptr[i].Depth = value
    @property
    def Depth(self):
        return self.get_Depth(0)
    @Depth.setter
    def Depth(self, int value):
        self.set_Depth(0,value)
    def get_Format(self, i):
        """Get value of ``Format`` of ``self._ptr[i]``.
        """
        return hipArray_Format(self._ptr[i].Format)
    def set_Format(self, i, value):
        """Set value ``Format`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].Format = value.value
    @property
    def Format(self):
        return self.get_Format(0)
    @Format.setter
    def Format(self, value):
        self.set_Format(0,value)
    def get_NumChannels(self, i):
        """Get value ``NumChannels`` of ``self._ptr[i]``.
        """
        return self._ptr[i].NumChannels
    def set_NumChannels(self, i, unsigned int value):
        """Set value ``NumChannels`` of ``self._ptr[i]``.
        """
        self._ptr[i].NumChannels = value
    @property
    def NumChannels(self):
        return self.get_NumChannels(0)
    @NumChannels.setter
    def NumChannels(self, unsigned int value):
        self.set_NumChannels(0,value)
    def get_Flags(self, i):
        """Get value ``Flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].Flags
    def set_Flags(self, i, unsigned int value):
        """Set value ``Flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].Flags = value
    @property
    def Flags(self):
        return self.get_Flags(0)
    @Flags.setter
    def Flags(self, unsigned int value):
        self.set_Flags(0,value)



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
    def get_desc(self, i):
        """Get value of ``desc`` of ``self._ptr[i]``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].desc)
    @property
    def desc(self):
        return self.get_desc(0)
    def get_type(self, i):
        """Get value ``type`` of ``self._ptr[i]``.
        """
        return self._ptr[i].type
    def set_type(self, i, unsigned int value):
        """Set value ``type`` of ``self._ptr[i]``.
        """
        self._ptr[i].type = value
    @property
    def type(self):
        return self.get_type(0)
    @type.setter
    def type(self, unsigned int value):
        self.set_type(0,value)
    def get_width(self, i):
        """Get value ``width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].width
    def set_width(self, i, unsigned int value):
        """Set value ``width`` of ``self._ptr[i]``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        return self.get_width(0)
    @width.setter
    def width(self, unsigned int value):
        self.set_width(0,value)
    def get_height(self, i):
        """Get value ``height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].height
    def set_height(self, i, unsigned int value):
        """Set value ``height`` of ``self._ptr[i]``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        return self.get_height(0)
    @height.setter
    def height(self, unsigned int value):
        self.set_height(0,value)
    def get_depth(self, i):
        """Get value ``depth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, unsigned int value):
        """Set value ``depth`` of ``self._ptr[i]``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        return self.get_depth(0)
    @depth.setter
    def depth(self, unsigned int value):
        self.set_depth(0,value)
    def get_Format(self, i):
        """Get value of ``Format`` of ``self._ptr[i]``.
        """
        return hipArray_Format(self._ptr[i].Format)
    def set_Format(self, i, value):
        """Set value ``Format`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].Format = value.value
    @property
    def Format(self):
        return self.get_Format(0)
    @Format.setter
    def Format(self, value):
        self.set_Format(0,value)
    def get_NumChannels(self, i):
        """Get value ``NumChannels`` of ``self._ptr[i]``.
        """
        return self._ptr[i].NumChannels
    def set_NumChannels(self, i, unsigned int value):
        """Set value ``NumChannels`` of ``self._ptr[i]``.
        """
        self._ptr[i].NumChannels = value
    @property
    def NumChannels(self):
        return self.get_NumChannels(0)
    @NumChannels.setter
    def NumChannels(self, unsigned int value):
        self.set_NumChannels(0,value)
    def get_isDrv(self, i):
        """Get value ``isDrv`` of ``self._ptr[i]``.
        """
        return self._ptr[i].isDrv
    def set_isDrv(self, i, int value):
        """Set value ``isDrv`` of ``self._ptr[i]``.
        """
        self._ptr[i].isDrv = value
    @property
    def isDrv(self):
        return self.get_isDrv(0)
    @isDrv.setter
    def isDrv(self, int value):
        self.set_isDrv(0,value)
    def get_textureType(self, i):
        """Get value ``textureType`` of ``self._ptr[i]``.
        """
        return self._ptr[i].textureType
    def set_textureType(self, i, unsigned int value):
        """Set value ``textureType`` of ``self._ptr[i]``.
        """
        self._ptr[i].textureType = value
    @property
    def textureType(self):
        return self.get_textureType(0)
    @textureType.setter
    def textureType(self, unsigned int value):
        self.set_textureType(0,value)



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
    def get_srcXInBytes(self, i):
        """Get value ``srcXInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].srcXInBytes
    def set_srcXInBytes(self, i, int value):
        """Set value ``srcXInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].srcXInBytes = value
    @property
    def srcXInBytes(self):
        return self.get_srcXInBytes(0)
    @srcXInBytes.setter
    def srcXInBytes(self, int value):
        self.set_srcXInBytes(0,value)
    def get_srcY(self, i):
        """Get value ``srcY`` of ``self._ptr[i]``.
        """
        return self._ptr[i].srcY
    def set_srcY(self, i, int value):
        """Set value ``srcY`` of ``self._ptr[i]``.
        """
        self._ptr[i].srcY = value
    @property
    def srcY(self):
        return self.get_srcY(0)
    @srcY.setter
    def srcY(self, int value):
        self.set_srcY(0,value)
    def get_srcMemoryType(self, i):
        """Get value of ``srcMemoryType`` of ``self._ptr[i]``.
        """
        return hipMemoryType(self._ptr[i].srcMemoryType)
    def set_srcMemoryType(self, i, value):
        """Set value ``srcMemoryType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].srcMemoryType = value.value
    @property
    def srcMemoryType(self):
        return self.get_srcMemoryType(0)
    @srcMemoryType.setter
    def srcMemoryType(self, value):
        self.set_srcMemoryType(0,value)
    def get_srcPitch(self, i):
        """Get value ``srcPitch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].srcPitch
    def set_srcPitch(self, i, int value):
        """Set value ``srcPitch`` of ``self._ptr[i]``.
        """
        self._ptr[i].srcPitch = value
    @property
    def srcPitch(self):
        return self.get_srcPitch(0)
    @srcPitch.setter
    def srcPitch(self, int value):
        self.set_srcPitch(0,value)
    def get_dstXInBytes(self, i):
        """Get value ``dstXInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].dstXInBytes
    def set_dstXInBytes(self, i, int value):
        """Set value ``dstXInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].dstXInBytes = value
    @property
    def dstXInBytes(self):
        return self.get_dstXInBytes(0)
    @dstXInBytes.setter
    def dstXInBytes(self, int value):
        self.set_dstXInBytes(0,value)
    def get_dstY(self, i):
        """Get value ``dstY`` of ``self._ptr[i]``.
        """
        return self._ptr[i].dstY
    def set_dstY(self, i, int value):
        """Set value ``dstY`` of ``self._ptr[i]``.
        """
        self._ptr[i].dstY = value
    @property
    def dstY(self):
        return self.get_dstY(0)
    @dstY.setter
    def dstY(self, int value):
        self.set_dstY(0,value)
    def get_dstMemoryType(self, i):
        """Get value of ``dstMemoryType`` of ``self._ptr[i]``.
        """
        return hipMemoryType(self._ptr[i].dstMemoryType)
    def set_dstMemoryType(self, i, value):
        """Set value ``dstMemoryType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].dstMemoryType = value.value
    @property
    def dstMemoryType(self):
        return self.get_dstMemoryType(0)
    @dstMemoryType.setter
    def dstMemoryType(self, value):
        self.set_dstMemoryType(0,value)
    def get_dstPitch(self, i):
        """Get value ``dstPitch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].dstPitch
    def set_dstPitch(self, i, int value):
        """Set value ``dstPitch`` of ``self._ptr[i]``.
        """
        self._ptr[i].dstPitch = value
    @property
    def dstPitch(self):
        return self.get_dstPitch(0)
    @dstPitch.setter
    def dstPitch(self, int value):
        self.set_dstPitch(0,value)
    def get_WidthInBytes(self, i):
        """Get value ``WidthInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].WidthInBytes
    def set_WidthInBytes(self, i, int value):
        """Set value ``WidthInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].WidthInBytes = value
    @property
    def WidthInBytes(self):
        return self.get_WidthInBytes(0)
    @WidthInBytes.setter
    def WidthInBytes(self, int value):
        self.set_WidthInBytes(0,value)
    def get_Height(self, i):
        """Get value ``Height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].Height
    def set_Height(self, i, int value):
        """Set value ``Height`` of ``self._ptr[i]``.
        """
        self._ptr[i].Height = value
    @property
    def Height(self):
        return self.get_Height(0)
    @Height.setter
    def Height(self, int value):
        self.set_Height(0,value)


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
    def get_desc(self, i):
        """Get value of ``desc`` of ``self._ptr[i]``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].desc)
    @property
    def desc(self):
        return self.get_desc(0)
    def get_type(self, i):
        """Get value ``type`` of ``self._ptr[i]``.
        """
        return self._ptr[i].type
    def set_type(self, i, unsigned int value):
        """Set value ``type`` of ``self._ptr[i]``.
        """
        self._ptr[i].type = value
    @property
    def type(self):
        return self.get_type(0)
    @type.setter
    def type(self, unsigned int value):
        self.set_type(0,value)
    def get_width(self, i):
        """Get value ``width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].width
    def set_width(self, i, unsigned int value):
        """Set value ``width`` of ``self._ptr[i]``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        return self.get_width(0)
    @width.setter
    def width(self, unsigned int value):
        self.set_width(0,value)
    def get_height(self, i):
        """Get value ``height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].height
    def set_height(self, i, unsigned int value):
        """Set value ``height`` of ``self._ptr[i]``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        return self.get_height(0)
    @height.setter
    def height(self, unsigned int value):
        self.set_height(0,value)
    def get_depth(self, i):
        """Get value ``depth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, unsigned int value):
        """Set value ``depth`` of ``self._ptr[i]``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        return self.get_depth(0)
    @depth.setter
    def depth(self, unsigned int value):
        self.set_depth(0,value)
    def get_min_mipmap_level(self, i):
        """Get value ``min_mipmap_level`` of ``self._ptr[i]``.
        """
        return self._ptr[i].min_mipmap_level
    def set_min_mipmap_level(self, i, unsigned int value):
        """Set value ``min_mipmap_level`` of ``self._ptr[i]``.
        """
        self._ptr[i].min_mipmap_level = value
    @property
    def min_mipmap_level(self):
        return self.get_min_mipmap_level(0)
    @min_mipmap_level.setter
    def min_mipmap_level(self, unsigned int value):
        self.set_min_mipmap_level(0,value)
    def get_max_mipmap_level(self, i):
        """Get value ``max_mipmap_level`` of ``self._ptr[i]``.
        """
        return self._ptr[i].max_mipmap_level
    def set_max_mipmap_level(self, i, unsigned int value):
        """Set value ``max_mipmap_level`` of ``self._ptr[i]``.
        """
        self._ptr[i].max_mipmap_level = value
    @property
    def max_mipmap_level(self):
        return self.get_max_mipmap_level(0)
    @max_mipmap_level.setter
    def max_mipmap_level(self, unsigned int value):
        self.set_max_mipmap_level(0,value)
    def get_flags(self, i):
        """Get value ``flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        self.set_flags(0,value)
    def get_format(self, i):
        """Get value of ``format`` of ``self._ptr[i]``.
        """
        return hipArray_Format(self._ptr[i].format)
    def set_format(self, i, value):
        """Set value ``format`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        return self.get_format(0)
    @format.setter
    def format(self, value):
        self.set_format(0,value)


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


cdef class HIPresourcetype:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIPresourcetype from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``HIPresourcetype`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIPresourcetype wrapper = HIPresourcetype.__new__(HIPresourcetype)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class hipResourcetype:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipResourcetype from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipResourcetype`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipResourcetype wrapper = hipResourcetype.__new__(hipResourcetype)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


class HIPaddress_mode_enum(enum.IntEnum):
    HIP_TR_ADDRESS_MODE_WRAP = chip.HIP_TR_ADDRESS_MODE_WRAP
    HIP_TR_ADDRESS_MODE_CLAMP = chip.HIP_TR_ADDRESS_MODE_CLAMP
    HIP_TR_ADDRESS_MODE_MIRROR = chip.HIP_TR_ADDRESS_MODE_MIRROR
    HIP_TR_ADDRESS_MODE_BORDER = chip.HIP_TR_ADDRESS_MODE_BORDER


cdef class HIPaddress_mode:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIPaddress_mode from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``HIPaddress_mode`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIPaddress_mode wrapper = HIPaddress_mode.__new__(HIPaddress_mode)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


class HIPfilter_mode_enum(enum.IntEnum):
    HIP_TR_FILTER_MODE_POINT = chip.HIP_TR_FILTER_MODE_POINT
    HIP_TR_FILTER_MODE_LINEAR = chip.HIP_TR_FILTER_MODE_LINEAR


cdef class HIPfilter_mode:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIPfilter_mode from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``HIPfilter_mode`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIPfilter_mode wrapper = HIPfilter_mode.__new__(HIPfilter_mode)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    # TODO is_enum_constantarray: add
    def get_filterMode(self, i):
        """Get value of ``filterMode`` of ``self._ptr[i]``.
        """
        return HIPfilter_mode_enum(self._ptr[i].filterMode)
    def set_filterMode(self, i, value):
        """Set value ``filterMode`` of ``self._ptr[i]``.
        """
        if not isinstance(value, HIPfilter_mode_enum):
            raise TypeError("'value' must be of type 'HIPfilter_mode_enum'")
        self._ptr[i].filterMode = value.value
    @property
    def filterMode(self):
        return self.get_filterMode(0)
    @filterMode.setter
    def filterMode(self, value):
        self.set_filterMode(0,value)
    def get_flags(self, i):
        """Get value ``flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        self.set_flags(0,value)
    def get_maxAnisotropy(self, i):
        """Get value ``maxAnisotropy`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxAnisotropy
    def set_maxAnisotropy(self, i, unsigned int value):
        """Set value ``maxAnisotropy`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxAnisotropy = value
    @property
    def maxAnisotropy(self):
        return self.get_maxAnisotropy(0)
    @maxAnisotropy.setter
    def maxAnisotropy(self, unsigned int value):
        self.set_maxAnisotropy(0,value)
    def get_mipmapFilterMode(self, i):
        """Get value of ``mipmapFilterMode`` of ``self._ptr[i]``.
        """
        return HIPfilter_mode_enum(self._ptr[i].mipmapFilterMode)
    def set_mipmapFilterMode(self, i, value):
        """Set value ``mipmapFilterMode`` of ``self._ptr[i]``.
        """
        if not isinstance(value, HIPfilter_mode_enum):
            raise TypeError("'value' must be of type 'HIPfilter_mode_enum'")
        self._ptr[i].mipmapFilterMode = value.value
    @property
    def mipmapFilterMode(self):
        return self.get_mipmapFilterMode(0)
    @mipmapFilterMode.setter
    def mipmapFilterMode(self, value):
        self.set_mipmapFilterMode(0,value)
    def get_mipmapLevelBias(self, i):
        """Get value ``mipmapLevelBias`` of ``self._ptr[i]``.
        """
        return self._ptr[i].mipmapLevelBias
    def set_mipmapLevelBias(self, i, float value):
        """Set value ``mipmapLevelBias`` of ``self._ptr[i]``.
        """
        self._ptr[i].mipmapLevelBias = value
    @property
    def mipmapLevelBias(self):
        return self.get_mipmapLevelBias(0)
    @mipmapLevelBias.setter
    def mipmapLevelBias(self, float value):
        self.set_mipmapLevelBias(0,value)
    def get_minMipmapLevelClamp(self, i):
        """Get value ``minMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        return self._ptr[i].minMipmapLevelClamp
    def set_minMipmapLevelClamp(self, i, float value):
        """Set value ``minMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        self._ptr[i].minMipmapLevelClamp = value
    @property
    def minMipmapLevelClamp(self):
        return self.get_minMipmapLevelClamp(0)
    @minMipmapLevelClamp.setter
    def minMipmapLevelClamp(self, float value):
        self.set_minMipmapLevelClamp(0,value)
    def get_maxMipmapLevelClamp(self, i):
        """Get value ``maxMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxMipmapLevelClamp
    def set_maxMipmapLevelClamp(self, i, float value):
        """Set value ``maxMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxMipmapLevelClamp = value
    @property
    def maxMipmapLevelClamp(self):
        return self.get_maxMipmapLevelClamp(0)
    @maxMipmapLevelClamp.setter
    def maxMipmapLevelClamp(self, float value):
        self.set_maxMipmapLevelClamp(0,value)
    def get_borderColor(self, i):
        """Get value of ``borderColor`` of ``self._ptr[i]``.
        """
        return self._ptr[i].borderColor
    @property
    def borderColor(self):
        return self.get_borderColor(0)
    # TODO is_basic_type_constantarray: add setters
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



cdef class HIP_TEXTURE_DESC:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_TEXTURE_DESC from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``HIP_TEXTURE_DESC`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_TEXTURE_DESC wrapper = HIP_TEXTURE_DESC.__new__(HIP_TEXTURE_DESC)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


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


cdef class HIPresourceViewFormat:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIPresourceViewFormat from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``HIPresourceViewFormat`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIPresourceViewFormat wrapper = HIPresourceViewFormat.__new__(HIPresourceViewFormat)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_desc(self, i):
        """Get value of ``desc`` of ``self._ptr[i]``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].desc)
    @property
    def desc(self):
        return self.get_desc(0)
    def get_sizeInBytes(self, i):
        """Get value ``sizeInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].sizeInBytes
    def set_sizeInBytes(self, i, int value):
        """Set value ``sizeInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].sizeInBytes = value
    @property
    def sizeInBytes(self):
        return self.get_sizeInBytes(0)
    @sizeInBytes.setter
    def sizeInBytes(self, int value):
        self.set_sizeInBytes(0,value)



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
    def get_desc(self, i):
        """Get value of ``desc`` of ``self._ptr[i]``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].desc)
    @property
    def desc(self):
        return self.get_desc(0)
    def get_width(self, i):
        """Get value ``width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Set value ``width`` of ``self._ptr[i]``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        self.set_width(0,value)
    def get_height(self, i):
        """Get value ``height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Set value ``height`` of ``self._ptr[i]``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        self.set_height(0,value)
    def get_pitchInBytes(self, i):
        """Get value ``pitchInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].pitchInBytes
    def set_pitchInBytes(self, i, int value):
        """Set value ``pitchInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].pitchInBytes = value
    @property
    def pitchInBytes(self):
        return self.get_pitchInBytes(0)
    @pitchInBytes.setter
    def pitchInBytes(self, int value):
        self.set_pitchInBytes(0,value)



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
    def get_array(self, i):
        """Get value of ``array`` of ``self._ptr[i]``.
        """
        return hipResourceDesc_union_0_struct_0.from_ptr(&self._ptr[i].array)
    @property
    def array(self):
        return self.get_array(0)
    def get_mipmap(self, i):
        """Get value of ``mipmap`` of ``self._ptr[i]``.
        """
        return hipResourceDesc_union_0_struct_1.from_ptr(&self._ptr[i].mipmap)
    @property
    def mipmap(self):
        return self.get_mipmap(0)
    def get_linear(self, i):
        """Get value of ``linear`` of ``self._ptr[i]``.
        """
        return hipResourceDesc_union_0_struct_2.from_ptr(&self._ptr[i].linear)
    @property
    def linear(self):
        return self.get_linear(0)
    def get_pitch2D(self, i):
        """Get value of ``pitch2D`` of ``self._ptr[i]``.
        """
        return hipResourceDesc_union_0_struct_3.from_ptr(&self._ptr[i].pitch2D)
    @property
    def pitch2D(self):
        return self.get_pitch2D(0)



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
    def get_resType(self, i):
        """Get value of ``resType`` of ``self._ptr[i]``.
        """
        return hipResourceType(self._ptr[i].resType)
    def set_resType(self, i, value):
        """Set value ``resType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipResourceType):
            raise TypeError("'value' must be of type 'hipResourceType'")
        self._ptr[i].resType = value.value
    @property
    def resType(self):
        return self.get_resType(0)
    @resType.setter
    def resType(self, value):
        self.set_resType(0,value)
    def get_res(self, i):
        """Get value of ``res`` of ``self._ptr[i]``.
        """
        return hipResourceDesc_union_0.from_ptr(&self._ptr[i].res)
    @property
    def res(self):
        return self.get_res(0)



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
    def get_format(self, i):
        """Get value of ``format`` of ``self._ptr[i]``.
        """
        return hipArray_Format(self._ptr[i].format)
    def set_format(self, i, value):
        """Set value ``format`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        return self.get_format(0)
    @format.setter
    def format(self, value):
        self.set_format(0,value)
    def get_numChannels(self, i):
        """Get value ``numChannels`` of ``self._ptr[i]``.
        """
        return self._ptr[i].numChannels
    def set_numChannels(self, i, unsigned int value):
        """Set value ``numChannels`` of ``self._ptr[i]``.
        """
        self._ptr[i].numChannels = value
    @property
    def numChannels(self):
        return self.get_numChannels(0)
    @numChannels.setter
    def numChannels(self, unsigned int value):
        self.set_numChannels(0,value)
    def get_sizeInBytes(self, i):
        """Get value ``sizeInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].sizeInBytes
    def set_sizeInBytes(self, i, int value):
        """Set value ``sizeInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].sizeInBytes = value
    @property
    def sizeInBytes(self):
        return self.get_sizeInBytes(0)
    @sizeInBytes.setter
    def sizeInBytes(self, int value):
        self.set_sizeInBytes(0,value)



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
    def get_format(self, i):
        """Get value of ``format`` of ``self._ptr[i]``.
        """
        return hipArray_Format(self._ptr[i].format)
    def set_format(self, i, value):
        """Set value ``format`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        return self.get_format(0)
    @format.setter
    def format(self, value):
        self.set_format(0,value)
    def get_numChannels(self, i):
        """Get value ``numChannels`` of ``self._ptr[i]``.
        """
        return self._ptr[i].numChannels
    def set_numChannels(self, i, unsigned int value):
        """Set value ``numChannels`` of ``self._ptr[i]``.
        """
        self._ptr[i].numChannels = value
    @property
    def numChannels(self):
        return self.get_numChannels(0)
    @numChannels.setter
    def numChannels(self, unsigned int value):
        self.set_numChannels(0,value)
    def get_width(self, i):
        """Get value ``width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Set value ``width`` of ``self._ptr[i]``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        self.set_width(0,value)
    def get_height(self, i):
        """Get value ``height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Set value ``height`` of ``self._ptr[i]``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        self.set_height(0,value)
    def get_pitchInBytes(self, i):
        """Get value ``pitchInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].pitchInBytes
    def set_pitchInBytes(self, i, int value):
        """Set value ``pitchInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].pitchInBytes = value
    @property
    def pitchInBytes(self):
        return self.get_pitchInBytes(0)
    @pitchInBytes.setter
    def pitchInBytes(self, int value):
        self.set_pitchInBytes(0,value)



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
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



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
    def get_array(self, i):
        """Get value of ``array`` of ``self._ptr[i]``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_0.from_ptr(&self._ptr[i].array)
    @property
    def array(self):
        return self.get_array(0)
    def get_mipmap(self, i):
        """Get value of ``mipmap`` of ``self._ptr[i]``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_1.from_ptr(&self._ptr[i].mipmap)
    @property
    def mipmap(self):
        return self.get_mipmap(0)
    def get_linear(self, i):
        """Get value of ``linear`` of ``self._ptr[i]``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_2.from_ptr(&self._ptr[i].linear)
    @property
    def linear(self):
        return self.get_linear(0)
    def get_pitch2D(self, i):
        """Get value of ``pitch2D`` of ``self._ptr[i]``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_3.from_ptr(&self._ptr[i].pitch2D)
    @property
    def pitch2D(self):
        return self.get_pitch2D(0)
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_4.from_ptr(&self._ptr[i].reserved)
    @property
    def reserved(self):
        return self.get_reserved(0)



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
    def get_resType(self, i):
        """Get value of ``resType`` of ``self._ptr[i]``.
        """
        return HIPresourcetype_enum(self._ptr[i].resType)
    def set_resType(self, i, value):
        """Set value ``resType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, HIPresourcetype_enum):
            raise TypeError("'value' must be of type 'HIPresourcetype_enum'")
        self._ptr[i].resType = value.value
    @property
    def resType(self):
        return self.get_resType(0)
    @resType.setter
    def resType(self, value):
        self.set_resType(0,value)
    def get_res(self, i):
        """Get value of ``res`` of ``self._ptr[i]``.
        """
        return HIP_RESOURCE_DESC_st_union_0.from_ptr(&self._ptr[i].res)
    @property
    def res(self):
        return self.get_res(0)
    def get_flags(self, i):
        """Get value ``flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        self.set_flags(0,value)



cdef class HIP_RESOURCE_DESC:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_DESC from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_DESC`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_DESC wrapper = HIP_RESOURCE_DESC.__new__(HIP_RESOURCE_DESC)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_format(self, i):
        """Get value of ``format`` of ``self._ptr[i]``.
        """
        return hipResourceViewFormat(self._ptr[i].format)
    def set_format(self, i, value):
        """Set value ``format`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipResourceViewFormat):
            raise TypeError("'value' must be of type 'hipResourceViewFormat'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        return self.get_format(0)
    @format.setter
    def format(self, value):
        self.set_format(0,value)
    def get_width(self, i):
        """Get value ``width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Set value ``width`` of ``self._ptr[i]``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        self.set_width(0,value)
    def get_height(self, i):
        """Get value ``height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Set value ``height`` of ``self._ptr[i]``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        self.set_height(0,value)
    def get_depth(self, i):
        """Get value ``depth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, int value):
        """Set value ``depth`` of ``self._ptr[i]``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        return self.get_depth(0)
    @depth.setter
    def depth(self, int value):
        self.set_depth(0,value)
    def get_firstMipmapLevel(self, i):
        """Get value ``firstMipmapLevel`` of ``self._ptr[i]``.
        """
        return self._ptr[i].firstMipmapLevel
    def set_firstMipmapLevel(self, i, unsigned int value):
        """Set value ``firstMipmapLevel`` of ``self._ptr[i]``.
        """
        self._ptr[i].firstMipmapLevel = value
    @property
    def firstMipmapLevel(self):
        return self.get_firstMipmapLevel(0)
    @firstMipmapLevel.setter
    def firstMipmapLevel(self, unsigned int value):
        self.set_firstMipmapLevel(0,value)
    def get_lastMipmapLevel(self, i):
        """Get value ``lastMipmapLevel`` of ``self._ptr[i]``.
        """
        return self._ptr[i].lastMipmapLevel
    def set_lastMipmapLevel(self, i, unsigned int value):
        """Set value ``lastMipmapLevel`` of ``self._ptr[i]``.
        """
        self._ptr[i].lastMipmapLevel = value
    @property
    def lastMipmapLevel(self):
        return self.get_lastMipmapLevel(0)
    @lastMipmapLevel.setter
    def lastMipmapLevel(self, unsigned int value):
        self.set_lastMipmapLevel(0,value)
    def get_firstLayer(self, i):
        """Get value ``firstLayer`` of ``self._ptr[i]``.
        """
        return self._ptr[i].firstLayer
    def set_firstLayer(self, i, unsigned int value):
        """Set value ``firstLayer`` of ``self._ptr[i]``.
        """
        self._ptr[i].firstLayer = value
    @property
    def firstLayer(self):
        return self.get_firstLayer(0)
    @firstLayer.setter
    def firstLayer(self, unsigned int value):
        self.set_firstLayer(0,value)
    def get_lastLayer(self, i):
        """Get value ``lastLayer`` of ``self._ptr[i]``.
        """
        return self._ptr[i].lastLayer
    def set_lastLayer(self, i, unsigned int value):
        """Set value ``lastLayer`` of ``self._ptr[i]``.
        """
        self._ptr[i].lastLayer = value
    @property
    def lastLayer(self):
        return self.get_lastLayer(0)
    @lastLayer.setter
    def lastLayer(self, unsigned int value):
        self.set_lastLayer(0,value)



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
    def get_format(self, i):
        """Get value of ``format`` of ``self._ptr[i]``.
        """
        return HIPresourceViewFormat_enum(self._ptr[i].format)
    def set_format(self, i, value):
        """Set value ``format`` of ``self._ptr[i]``.
        """
        if not isinstance(value, HIPresourceViewFormat_enum):
            raise TypeError("'value' must be of type 'HIPresourceViewFormat_enum'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        return self.get_format(0)
    @format.setter
    def format(self, value):
        self.set_format(0,value)
    def get_width(self, i):
        """Get value ``width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Set value ``width`` of ``self._ptr[i]``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        self.set_width(0,value)
    def get_height(self, i):
        """Get value ``height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Set value ``height`` of ``self._ptr[i]``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        self.set_height(0,value)
    def get_depth(self, i):
        """Get value ``depth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, int value):
        """Set value ``depth`` of ``self._ptr[i]``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        return self.get_depth(0)
    @depth.setter
    def depth(self, int value):
        self.set_depth(0,value)
    def get_firstMipmapLevel(self, i):
        """Get value ``firstMipmapLevel`` of ``self._ptr[i]``.
        """
        return self._ptr[i].firstMipmapLevel
    def set_firstMipmapLevel(self, i, unsigned int value):
        """Set value ``firstMipmapLevel`` of ``self._ptr[i]``.
        """
        self._ptr[i].firstMipmapLevel = value
    @property
    def firstMipmapLevel(self):
        return self.get_firstMipmapLevel(0)
    @firstMipmapLevel.setter
    def firstMipmapLevel(self, unsigned int value):
        self.set_firstMipmapLevel(0,value)
    def get_lastMipmapLevel(self, i):
        """Get value ``lastMipmapLevel`` of ``self._ptr[i]``.
        """
        return self._ptr[i].lastMipmapLevel
    def set_lastMipmapLevel(self, i, unsigned int value):
        """Set value ``lastMipmapLevel`` of ``self._ptr[i]``.
        """
        self._ptr[i].lastMipmapLevel = value
    @property
    def lastMipmapLevel(self):
        return self.get_lastMipmapLevel(0)
    @lastMipmapLevel.setter
    def lastMipmapLevel(self, unsigned int value):
        self.set_lastMipmapLevel(0,value)
    def get_firstLayer(self, i):
        """Get value ``firstLayer`` of ``self._ptr[i]``.
        """
        return self._ptr[i].firstLayer
    def set_firstLayer(self, i, unsigned int value):
        """Set value ``firstLayer`` of ``self._ptr[i]``.
        """
        self._ptr[i].firstLayer = value
    @property
    def firstLayer(self):
        return self.get_firstLayer(0)
    @firstLayer.setter
    def firstLayer(self, unsigned int value):
        self.set_firstLayer(0,value)
    def get_lastLayer(self, i):
        """Get value ``lastLayer`` of ``self._ptr[i]``.
        """
        return self._ptr[i].lastLayer
    def set_lastLayer(self, i, unsigned int value):
        """Set value ``lastLayer`` of ``self._ptr[i]``.
        """
        self._ptr[i].lastLayer = value
    @property
    def lastLayer(self):
        return self.get_lastLayer(0)
    @lastLayer.setter
    def lastLayer(self, unsigned int value):
        self.set_lastLayer(0,value)
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



cdef class HIP_RESOURCE_VIEW_DESC:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef HIP_RESOURCE_VIEW_DESC from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``HIP_RESOURCE_VIEW_DESC`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef HIP_RESOURCE_VIEW_DESC wrapper = HIP_RESOURCE_VIEW_DESC.__new__(HIP_RESOURCE_VIEW_DESC)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


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
    def get_pitch(self, i):
        """Get value ``pitch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].pitch
    def set_pitch(self, i, int value):
        """Set value ``pitch`` of ``self._ptr[i]``.
        """
        self._ptr[i].pitch = value
    @property
    def pitch(self):
        return self.get_pitch(0)
    @pitch.setter
    def pitch(self, int value):
        self.set_pitch(0,value)
    def get_xsize(self, i):
        """Get value ``xsize`` of ``self._ptr[i]``.
        """
        return self._ptr[i].xsize
    def set_xsize(self, i, int value):
        """Set value ``xsize`` of ``self._ptr[i]``.
        """
        self._ptr[i].xsize = value
    @property
    def xsize(self):
        return self.get_xsize(0)
    @xsize.setter
    def xsize(self, int value):
        self.set_xsize(0,value)
    def get_ysize(self, i):
        """Get value ``ysize`` of ``self._ptr[i]``.
        """
        return self._ptr[i].ysize
    def set_ysize(self, i, int value):
        """Set value ``ysize`` of ``self._ptr[i]``.
        """
        self._ptr[i].ysize = value
    @property
    def ysize(self):
        return self.get_ysize(0)
    @ysize.setter
    def ysize(self, int value):
        self.set_ysize(0,value)



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
    def get_width(self, i):
        """Get value ``width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Set value ``width`` of ``self._ptr[i]``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        self.set_width(0,value)
    def get_height(self, i):
        """Get value ``height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Set value ``height`` of ``self._ptr[i]``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        self.set_height(0,value)
    def get_depth(self, i):
        """Get value ``depth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, int value):
        """Set value ``depth`` of ``self._ptr[i]``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        return self.get_depth(0)
    @depth.setter
    def depth(self, int value):
        self.set_depth(0,value)



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
    def get_x(self, i):
        """Get value ``x`` of ``self._ptr[i]``.
        """
        return self._ptr[i].x
    def set_x(self, i, int value):
        """Set value ``x`` of ``self._ptr[i]``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        return self.get_x(0)
    @x.setter
    def x(self, int value):
        self.set_x(0,value)
    def get_y(self, i):
        """Get value ``y`` of ``self._ptr[i]``.
        """
        return self._ptr[i].y
    def set_y(self, i, int value):
        """Set value ``y`` of ``self._ptr[i]``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        return self.get_y(0)
    @y.setter
    def y(self, int value):
        self.set_y(0,value)
    def get_z(self, i):
        """Get value ``z`` of ``self._ptr[i]``.
        """
        return self._ptr[i].z
    def set_z(self, i, int value):
        """Set value ``z`` of ``self._ptr[i]``.
        """
        self._ptr[i].z = value
    @property
    def z(self):
        return self.get_z(0)
    @z.setter
    def z(self, int value):
        self.set_z(0,value)



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
    def get_srcPos(self, i):
        """Get value of ``srcPos`` of ``self._ptr[i]``.
        """
        return hipPos.from_ptr(&self._ptr[i].srcPos)
    @property
    def srcPos(self):
        return self.get_srcPos(0)
    def get_srcPtr(self, i):
        """Get value of ``srcPtr`` of ``self._ptr[i]``.
        """
        return hipPitchedPtr.from_ptr(&self._ptr[i].srcPtr)
    @property
    def srcPtr(self):
        return self.get_srcPtr(0)
    def get_dstPos(self, i):
        """Get value of ``dstPos`` of ``self._ptr[i]``.
        """
        return hipPos.from_ptr(&self._ptr[i].dstPos)
    @property
    def dstPos(self):
        return self.get_dstPos(0)
    def get_dstPtr(self, i):
        """Get value of ``dstPtr`` of ``self._ptr[i]``.
        """
        return hipPitchedPtr.from_ptr(&self._ptr[i].dstPtr)
    @property
    def dstPtr(self):
        return self.get_dstPtr(0)
    def get_extent(self, i):
        """Get value of ``extent`` of ``self._ptr[i]``.
        """
        return hipExtent.from_ptr(&self._ptr[i].extent)
    @property
    def extent(self):
        return self.get_extent(0)
    def get_kind(self, i):
        """Get value of ``kind`` of ``self._ptr[i]``.
        """
        return hipMemcpyKind(self._ptr[i].kind)
    def set_kind(self, i, value):
        """Set value ``kind`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemcpyKind):
            raise TypeError("'value' must be of type 'hipMemcpyKind'")
        self._ptr[i].kind = value.value
    @property
    def kind(self):
        return self.get_kind(0)
    @kind.setter
    def kind(self, value):
        self.set_kind(0,value)



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
    def get_srcXInBytes(self, i):
        """Get value ``srcXInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].srcXInBytes
    def set_srcXInBytes(self, i, unsigned int value):
        """Set value ``srcXInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].srcXInBytes = value
    @property
    def srcXInBytes(self):
        return self.get_srcXInBytes(0)
    @srcXInBytes.setter
    def srcXInBytes(self, unsigned int value):
        self.set_srcXInBytes(0,value)
    def get_srcY(self, i):
        """Get value ``srcY`` of ``self._ptr[i]``.
        """
        return self._ptr[i].srcY
    def set_srcY(self, i, unsigned int value):
        """Set value ``srcY`` of ``self._ptr[i]``.
        """
        self._ptr[i].srcY = value
    @property
    def srcY(self):
        return self.get_srcY(0)
    @srcY.setter
    def srcY(self, unsigned int value):
        self.set_srcY(0,value)
    def get_srcZ(self, i):
        """Get value ``srcZ`` of ``self._ptr[i]``.
        """
        return self._ptr[i].srcZ
    def set_srcZ(self, i, unsigned int value):
        """Set value ``srcZ`` of ``self._ptr[i]``.
        """
        self._ptr[i].srcZ = value
    @property
    def srcZ(self):
        return self.get_srcZ(0)
    @srcZ.setter
    def srcZ(self, unsigned int value):
        self.set_srcZ(0,value)
    def get_srcLOD(self, i):
        """Get value ``srcLOD`` of ``self._ptr[i]``.
        """
        return self._ptr[i].srcLOD
    def set_srcLOD(self, i, unsigned int value):
        """Set value ``srcLOD`` of ``self._ptr[i]``.
        """
        self._ptr[i].srcLOD = value
    @property
    def srcLOD(self):
        return self.get_srcLOD(0)
    @srcLOD.setter
    def srcLOD(self, unsigned int value):
        self.set_srcLOD(0,value)
    def get_srcMemoryType(self, i):
        """Get value of ``srcMemoryType`` of ``self._ptr[i]``.
        """
        return hipMemoryType(self._ptr[i].srcMemoryType)
    def set_srcMemoryType(self, i, value):
        """Set value ``srcMemoryType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].srcMemoryType = value.value
    @property
    def srcMemoryType(self):
        return self.get_srcMemoryType(0)
    @srcMemoryType.setter
    def srcMemoryType(self, value):
        self.set_srcMemoryType(0,value)
    def get_srcPitch(self, i):
        """Get value ``srcPitch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].srcPitch
    def set_srcPitch(self, i, unsigned int value):
        """Set value ``srcPitch`` of ``self._ptr[i]``.
        """
        self._ptr[i].srcPitch = value
    @property
    def srcPitch(self):
        return self.get_srcPitch(0)
    @srcPitch.setter
    def srcPitch(self, unsigned int value):
        self.set_srcPitch(0,value)
    def get_srcHeight(self, i):
        """Get value ``srcHeight`` of ``self._ptr[i]``.
        """
        return self._ptr[i].srcHeight
    def set_srcHeight(self, i, unsigned int value):
        """Set value ``srcHeight`` of ``self._ptr[i]``.
        """
        self._ptr[i].srcHeight = value
    @property
    def srcHeight(self):
        return self.get_srcHeight(0)
    @srcHeight.setter
    def srcHeight(self, unsigned int value):
        self.set_srcHeight(0,value)
    def get_dstXInBytes(self, i):
        """Get value ``dstXInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].dstXInBytes
    def set_dstXInBytes(self, i, unsigned int value):
        """Set value ``dstXInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].dstXInBytes = value
    @property
    def dstXInBytes(self):
        return self.get_dstXInBytes(0)
    @dstXInBytes.setter
    def dstXInBytes(self, unsigned int value):
        self.set_dstXInBytes(0,value)
    def get_dstY(self, i):
        """Get value ``dstY`` of ``self._ptr[i]``.
        """
        return self._ptr[i].dstY
    def set_dstY(self, i, unsigned int value):
        """Set value ``dstY`` of ``self._ptr[i]``.
        """
        self._ptr[i].dstY = value
    @property
    def dstY(self):
        return self.get_dstY(0)
    @dstY.setter
    def dstY(self, unsigned int value):
        self.set_dstY(0,value)
    def get_dstZ(self, i):
        """Get value ``dstZ`` of ``self._ptr[i]``.
        """
        return self._ptr[i].dstZ
    def set_dstZ(self, i, unsigned int value):
        """Set value ``dstZ`` of ``self._ptr[i]``.
        """
        self._ptr[i].dstZ = value
    @property
    def dstZ(self):
        return self.get_dstZ(0)
    @dstZ.setter
    def dstZ(self, unsigned int value):
        self.set_dstZ(0,value)
    def get_dstLOD(self, i):
        """Get value ``dstLOD`` of ``self._ptr[i]``.
        """
        return self._ptr[i].dstLOD
    def set_dstLOD(self, i, unsigned int value):
        """Set value ``dstLOD`` of ``self._ptr[i]``.
        """
        self._ptr[i].dstLOD = value
    @property
    def dstLOD(self):
        return self.get_dstLOD(0)
    @dstLOD.setter
    def dstLOD(self, unsigned int value):
        self.set_dstLOD(0,value)
    def get_dstMemoryType(self, i):
        """Get value of ``dstMemoryType`` of ``self._ptr[i]``.
        """
        return hipMemoryType(self._ptr[i].dstMemoryType)
    def set_dstMemoryType(self, i, value):
        """Set value ``dstMemoryType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].dstMemoryType = value.value
    @property
    def dstMemoryType(self):
        return self.get_dstMemoryType(0)
    @dstMemoryType.setter
    def dstMemoryType(self, value):
        self.set_dstMemoryType(0,value)
    def get_dstPitch(self, i):
        """Get value ``dstPitch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].dstPitch
    def set_dstPitch(self, i, unsigned int value):
        """Set value ``dstPitch`` of ``self._ptr[i]``.
        """
        self._ptr[i].dstPitch = value
    @property
    def dstPitch(self):
        return self.get_dstPitch(0)
    @dstPitch.setter
    def dstPitch(self, unsigned int value):
        self.set_dstPitch(0,value)
    def get_dstHeight(self, i):
        """Get value ``dstHeight`` of ``self._ptr[i]``.
        """
        return self._ptr[i].dstHeight
    def set_dstHeight(self, i, unsigned int value):
        """Set value ``dstHeight`` of ``self._ptr[i]``.
        """
        self._ptr[i].dstHeight = value
    @property
    def dstHeight(self):
        return self.get_dstHeight(0)
    @dstHeight.setter
    def dstHeight(self, unsigned int value):
        self.set_dstHeight(0,value)
    def get_WidthInBytes(self, i):
        """Get value ``WidthInBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].WidthInBytes
    def set_WidthInBytes(self, i, unsigned int value):
        """Set value ``WidthInBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].WidthInBytes = value
    @property
    def WidthInBytes(self):
        return self.get_WidthInBytes(0)
    @WidthInBytes.setter
    def WidthInBytes(self, unsigned int value):
        self.set_WidthInBytes(0,value)
    def get_Height(self, i):
        """Get value ``Height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].Height
    def set_Height(self, i, unsigned int value):
        """Set value ``Height`` of ``self._ptr[i]``.
        """
        self._ptr[i].Height = value
    @property
    def Height(self):
        return self.get_Height(0)
    @Height.setter
    def Height(self, unsigned int value):
        self.set_Height(0,value)
    def get_Depth(self, i):
        """Get value ``Depth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].Depth
    def set_Depth(self, i, unsigned int value):
        """Set value ``Depth`` of ``self._ptr[i]``.
        """
        self._ptr[i].Depth = value
    @property
    def Depth(self):
        return self.get_Depth(0)
    @Depth.setter
    def Depth(self, unsigned int value):
        self.set_Depth(0,value)


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


def hipCreateChannelDesc(int x, int y, int z, int w, f):
    """
    """
    pass


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
    def get_normalized(self, i):
        """Get value ``normalized`` of ``self._ptr[i]``.
        """
        return self._ptr[i].normalized
    def set_normalized(self, i, int value):
        """Set value ``normalized`` of ``self._ptr[i]``.
        """
        self._ptr[i].normalized = value
    @property
    def normalized(self):
        return self.get_normalized(0)
    @normalized.setter
    def normalized(self, int value):
        self.set_normalized(0,value)
    def get_readMode(self, i):
        """Get value of ``readMode`` of ``self._ptr[i]``.
        """
        return hipTextureReadMode(self._ptr[i].readMode)
    def set_readMode(self, i, value):
        """Set value ``readMode`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipTextureReadMode):
            raise TypeError("'value' must be of type 'hipTextureReadMode'")
        self._ptr[i].readMode = value.value
    @property
    def readMode(self):
        return self.get_readMode(0)
    @readMode.setter
    def readMode(self, value):
        self.set_readMode(0,value)
    def get_filterMode(self, i):
        """Get value of ``filterMode`` of ``self._ptr[i]``.
        """
        return hipTextureFilterMode(self._ptr[i].filterMode)
    def set_filterMode(self, i, value):
        """Set value ``filterMode`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipTextureFilterMode):
            raise TypeError("'value' must be of type 'hipTextureFilterMode'")
        self._ptr[i].filterMode = value.value
    @property
    def filterMode(self):
        return self.get_filterMode(0)
    @filterMode.setter
    def filterMode(self, value):
        self.set_filterMode(0,value)
    # TODO is_enum_constantarray: add
    def get_channelDesc(self, i):
        """Get value of ``channelDesc`` of ``self._ptr[i]``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].channelDesc)
    @property
    def channelDesc(self):
        return self.get_channelDesc(0)
    def get_sRGB(self, i):
        """Get value ``sRGB`` of ``self._ptr[i]``.
        """
        return self._ptr[i].sRGB
    def set_sRGB(self, i, int value):
        """Set value ``sRGB`` of ``self._ptr[i]``.
        """
        self._ptr[i].sRGB = value
    @property
    def sRGB(self):
        return self.get_sRGB(0)
    @sRGB.setter
    def sRGB(self, int value):
        self.set_sRGB(0,value)
    def get_maxAnisotropy(self, i):
        """Get value ``maxAnisotropy`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxAnisotropy
    def set_maxAnisotropy(self, i, unsigned int value):
        """Set value ``maxAnisotropy`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxAnisotropy = value
    @property
    def maxAnisotropy(self):
        return self.get_maxAnisotropy(0)
    @maxAnisotropy.setter
    def maxAnisotropy(self, unsigned int value):
        self.set_maxAnisotropy(0,value)
    def get_mipmapFilterMode(self, i):
        """Get value of ``mipmapFilterMode`` of ``self._ptr[i]``.
        """
        return hipTextureFilterMode(self._ptr[i].mipmapFilterMode)
    def set_mipmapFilterMode(self, i, value):
        """Set value ``mipmapFilterMode`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipTextureFilterMode):
            raise TypeError("'value' must be of type 'hipTextureFilterMode'")
        self._ptr[i].mipmapFilterMode = value.value
    @property
    def mipmapFilterMode(self):
        return self.get_mipmapFilterMode(0)
    @mipmapFilterMode.setter
    def mipmapFilterMode(self, value):
        self.set_mipmapFilterMode(0,value)
    def get_mipmapLevelBias(self, i):
        """Get value ``mipmapLevelBias`` of ``self._ptr[i]``.
        """
        return self._ptr[i].mipmapLevelBias
    def set_mipmapLevelBias(self, i, float value):
        """Set value ``mipmapLevelBias`` of ``self._ptr[i]``.
        """
        self._ptr[i].mipmapLevelBias = value
    @property
    def mipmapLevelBias(self):
        return self.get_mipmapLevelBias(0)
    @mipmapLevelBias.setter
    def mipmapLevelBias(self, float value):
        self.set_mipmapLevelBias(0,value)
    def get_minMipmapLevelClamp(self, i):
        """Get value ``minMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        return self._ptr[i].minMipmapLevelClamp
    def set_minMipmapLevelClamp(self, i, float value):
        """Set value ``minMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        self._ptr[i].minMipmapLevelClamp = value
    @property
    def minMipmapLevelClamp(self):
        return self.get_minMipmapLevelClamp(0)
    @minMipmapLevelClamp.setter
    def minMipmapLevelClamp(self, float value):
        self.set_minMipmapLevelClamp(0,value)
    def get_maxMipmapLevelClamp(self, i):
        """Get value ``maxMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxMipmapLevelClamp
    def set_maxMipmapLevelClamp(self, i, float value):
        """Set value ``maxMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxMipmapLevelClamp = value
    @property
    def maxMipmapLevelClamp(self):
        return self.get_maxMipmapLevelClamp(0)
    @maxMipmapLevelClamp.setter
    def maxMipmapLevelClamp(self, float value):
        self.set_maxMipmapLevelClamp(0,value)
    def get_numChannels(self, i):
        """Get value ``numChannels`` of ``self._ptr[i]``.
        """
        return self._ptr[i].numChannels
    def set_numChannels(self, i, int value):
        """Set value ``numChannels`` of ``self._ptr[i]``.
        """
        self._ptr[i].numChannels = value
    @property
    def numChannels(self):
        return self.get_numChannels(0)
    @numChannels.setter
    def numChannels(self, int value):
        self.set_numChannels(0,value)
    def get_format(self, i):
        """Get value of ``format`` of ``self._ptr[i]``.
        """
        return hipArray_Format(self._ptr[i].format)
    def set_format(self, i, value):
        """Set value ``format`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        return self.get_format(0)
    @format.setter
    def format(self, value):
        self.set_format(0,value)



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
    # TODO is_enum_constantarray: add
    def get_filterMode(self, i):
        """Get value of ``filterMode`` of ``self._ptr[i]``.
        """
        return hipTextureFilterMode(self._ptr[i].filterMode)
    def set_filterMode(self, i, value):
        """Set value ``filterMode`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipTextureFilterMode):
            raise TypeError("'value' must be of type 'hipTextureFilterMode'")
        self._ptr[i].filterMode = value.value
    @property
    def filterMode(self):
        return self.get_filterMode(0)
    @filterMode.setter
    def filterMode(self, value):
        self.set_filterMode(0,value)
    def get_readMode(self, i):
        """Get value of ``readMode`` of ``self._ptr[i]``.
        """
        return hipTextureReadMode(self._ptr[i].readMode)
    def set_readMode(self, i, value):
        """Set value ``readMode`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipTextureReadMode):
            raise TypeError("'value' must be of type 'hipTextureReadMode'")
        self._ptr[i].readMode = value.value
    @property
    def readMode(self):
        return self.get_readMode(0)
    @readMode.setter
    def readMode(self, value):
        self.set_readMode(0,value)
    def get_sRGB(self, i):
        """Get value ``sRGB`` of ``self._ptr[i]``.
        """
        return self._ptr[i].sRGB
    def set_sRGB(self, i, int value):
        """Set value ``sRGB`` of ``self._ptr[i]``.
        """
        self._ptr[i].sRGB = value
    @property
    def sRGB(self):
        return self.get_sRGB(0)
    @sRGB.setter
    def sRGB(self, int value):
        self.set_sRGB(0,value)
    def get_borderColor(self, i):
        """Get value of ``borderColor`` of ``self._ptr[i]``.
        """
        return self._ptr[i].borderColor
    @property
    def borderColor(self):
        return self.get_borderColor(0)
    # TODO is_basic_type_constantarray: add setters
    def get_normalizedCoords(self, i):
        """Get value ``normalizedCoords`` of ``self._ptr[i]``.
        """
        return self._ptr[i].normalizedCoords
    def set_normalizedCoords(self, i, int value):
        """Set value ``normalizedCoords`` of ``self._ptr[i]``.
        """
        self._ptr[i].normalizedCoords = value
    @property
    def normalizedCoords(self):
        return self.get_normalizedCoords(0)
    @normalizedCoords.setter
    def normalizedCoords(self, int value):
        self.set_normalizedCoords(0,value)
    def get_maxAnisotropy(self, i):
        """Get value ``maxAnisotropy`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxAnisotropy
    def set_maxAnisotropy(self, i, unsigned int value):
        """Set value ``maxAnisotropy`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxAnisotropy = value
    @property
    def maxAnisotropy(self):
        return self.get_maxAnisotropy(0)
    @maxAnisotropy.setter
    def maxAnisotropy(self, unsigned int value):
        self.set_maxAnisotropy(0,value)
    def get_mipmapFilterMode(self, i):
        """Get value of ``mipmapFilterMode`` of ``self._ptr[i]``.
        """
        return hipTextureFilterMode(self._ptr[i].mipmapFilterMode)
    def set_mipmapFilterMode(self, i, value):
        """Set value ``mipmapFilterMode`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipTextureFilterMode):
            raise TypeError("'value' must be of type 'hipTextureFilterMode'")
        self._ptr[i].mipmapFilterMode = value.value
    @property
    def mipmapFilterMode(self):
        return self.get_mipmapFilterMode(0)
    @mipmapFilterMode.setter
    def mipmapFilterMode(self, value):
        self.set_mipmapFilterMode(0,value)
    def get_mipmapLevelBias(self, i):
        """Get value ``mipmapLevelBias`` of ``self._ptr[i]``.
        """
        return self._ptr[i].mipmapLevelBias
    def set_mipmapLevelBias(self, i, float value):
        """Set value ``mipmapLevelBias`` of ``self._ptr[i]``.
        """
        self._ptr[i].mipmapLevelBias = value
    @property
    def mipmapLevelBias(self):
        return self.get_mipmapLevelBias(0)
    @mipmapLevelBias.setter
    def mipmapLevelBias(self, float value):
        self.set_mipmapLevelBias(0,value)
    def get_minMipmapLevelClamp(self, i):
        """Get value ``minMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        return self._ptr[i].minMipmapLevelClamp
    def set_minMipmapLevelClamp(self, i, float value):
        """Set value ``minMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        self._ptr[i].minMipmapLevelClamp = value
    @property
    def minMipmapLevelClamp(self):
        return self.get_minMipmapLevelClamp(0)
    @minMipmapLevelClamp.setter
    def minMipmapLevelClamp(self, float value):
        self.set_minMipmapLevelClamp(0,value)
    def get_maxMipmapLevelClamp(self, i):
        """Get value ``maxMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxMipmapLevelClamp
    def set_maxMipmapLevelClamp(self, i, float value):
        """Set value ``maxMipmapLevelClamp`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxMipmapLevelClamp = value
    @property
    def maxMipmapLevelClamp(self):
        return self.get_maxMipmapLevelClamp(0)
    @maxMipmapLevelClamp.setter
    def maxMipmapLevelClamp(self, float value):
        self.set_maxMipmapLevelClamp(0,value)



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


cdef class hipDevice_t:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipDevice_t from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipDevice_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipDevice_t wrapper = hipDevice_t.__new__(hipDevice_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


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
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



cdef class hipIpcMemHandle_t:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipIpcMemHandle_t from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipIpcMemHandle_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipIpcMemHandle_t wrapper = hipIpcMemHandle_t.__new__(hipIpcMemHandle_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



cdef class hipIpcEventHandle_t:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipIpcEventHandle_t from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipIpcEventHandle_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipIpcEventHandle_t wrapper = hipIpcEventHandle_t.__new__(hipIpcEventHandle_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_binaryVersion(self, i):
        """Get value ``binaryVersion`` of ``self._ptr[i]``.
        """
        return self._ptr[i].binaryVersion
    def set_binaryVersion(self, i, int value):
        """Set value ``binaryVersion`` of ``self._ptr[i]``.
        """
        self._ptr[i].binaryVersion = value
    @property
    def binaryVersion(self):
        return self.get_binaryVersion(0)
    @binaryVersion.setter
    def binaryVersion(self, int value):
        self.set_binaryVersion(0,value)
    def get_cacheModeCA(self, i):
        """Get value ``cacheModeCA`` of ``self._ptr[i]``.
        """
        return self._ptr[i].cacheModeCA
    def set_cacheModeCA(self, i, int value):
        """Set value ``cacheModeCA`` of ``self._ptr[i]``.
        """
        self._ptr[i].cacheModeCA = value
    @property
    def cacheModeCA(self):
        return self.get_cacheModeCA(0)
    @cacheModeCA.setter
    def cacheModeCA(self, int value):
        self.set_cacheModeCA(0,value)
    def get_constSizeBytes(self, i):
        """Get value ``constSizeBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].constSizeBytes
    def set_constSizeBytes(self, i, int value):
        """Set value ``constSizeBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].constSizeBytes = value
    @property
    def constSizeBytes(self):
        return self.get_constSizeBytes(0)
    @constSizeBytes.setter
    def constSizeBytes(self, int value):
        self.set_constSizeBytes(0,value)
    def get_localSizeBytes(self, i):
        """Get value ``localSizeBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].localSizeBytes
    def set_localSizeBytes(self, i, int value):
        """Set value ``localSizeBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].localSizeBytes = value
    @property
    def localSizeBytes(self):
        return self.get_localSizeBytes(0)
    @localSizeBytes.setter
    def localSizeBytes(self, int value):
        self.set_localSizeBytes(0,value)
    def get_maxDynamicSharedSizeBytes(self, i):
        """Get value ``maxDynamicSharedSizeBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxDynamicSharedSizeBytes
    def set_maxDynamicSharedSizeBytes(self, i, int value):
        """Set value ``maxDynamicSharedSizeBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxDynamicSharedSizeBytes = value
    @property
    def maxDynamicSharedSizeBytes(self):
        return self.get_maxDynamicSharedSizeBytes(0)
    @maxDynamicSharedSizeBytes.setter
    def maxDynamicSharedSizeBytes(self, int value):
        self.set_maxDynamicSharedSizeBytes(0,value)
    def get_maxThreadsPerBlock(self, i):
        """Get value ``maxThreadsPerBlock`` of ``self._ptr[i]``.
        """
        return self._ptr[i].maxThreadsPerBlock
    def set_maxThreadsPerBlock(self, i, int value):
        """Set value ``maxThreadsPerBlock`` of ``self._ptr[i]``.
        """
        self._ptr[i].maxThreadsPerBlock = value
    @property
    def maxThreadsPerBlock(self):
        return self.get_maxThreadsPerBlock(0)
    @maxThreadsPerBlock.setter
    def maxThreadsPerBlock(self, int value):
        self.set_maxThreadsPerBlock(0,value)
    def get_numRegs(self, i):
        """Get value ``numRegs`` of ``self._ptr[i]``.
        """
        return self._ptr[i].numRegs
    def set_numRegs(self, i, int value):
        """Set value ``numRegs`` of ``self._ptr[i]``.
        """
        self._ptr[i].numRegs = value
    @property
    def numRegs(self):
        return self.get_numRegs(0)
    @numRegs.setter
    def numRegs(self, int value):
        self.set_numRegs(0,value)
    def get_preferredShmemCarveout(self, i):
        """Get value ``preferredShmemCarveout`` of ``self._ptr[i]``.
        """
        return self._ptr[i].preferredShmemCarveout
    def set_preferredShmemCarveout(self, i, int value):
        """Set value ``preferredShmemCarveout`` of ``self._ptr[i]``.
        """
        self._ptr[i].preferredShmemCarveout = value
    @property
    def preferredShmemCarveout(self):
        return self.get_preferredShmemCarveout(0)
    @preferredShmemCarveout.setter
    def preferredShmemCarveout(self, int value):
        self.set_preferredShmemCarveout(0,value)
    def get_ptxVersion(self, i):
        """Get value ``ptxVersion`` of ``self._ptr[i]``.
        """
        return self._ptr[i].ptxVersion
    def set_ptxVersion(self, i, int value):
        """Set value ``ptxVersion`` of ``self._ptr[i]``.
        """
        self._ptr[i].ptxVersion = value
    @property
    def ptxVersion(self):
        return self.get_ptxVersion(0)
    @ptxVersion.setter
    def ptxVersion(self, int value):
        self.set_ptxVersion(0,value)
    def get_sharedSizeBytes(self, i):
        """Get value ``sharedSizeBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].sharedSizeBytes
    def set_sharedSizeBytes(self, i, int value):
        """Set value ``sharedSizeBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].sharedSizeBytes = value
    @property
    def sharedSizeBytes(self):
        return self.get_sharedSizeBytes(0)
    @sharedSizeBytes.setter
    def sharedSizeBytes(self, int value):
        self.set_sharedSizeBytes(0,value)



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
    def get_type(self, i):
        """Get value of ``type`` of ``self._ptr[i]``.
        """
        return hipMemLocationType(self._ptr[i].type)
    def set_type(self, i, value):
        """Set value ``type`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemLocationType):
            raise TypeError("'value' must be of type 'hipMemLocationType'")
        self._ptr[i].type = value.value
    @property
    def type(self):
        return self.get_type(0)
    @type.setter
    def type(self, value):
        self.set_type(0,value)
    def get_id(self, i):
        """Get value ``id`` of ``self._ptr[i]``.
        """
        return self._ptr[i].id
    def set_id(self, i, int value):
        """Set value ``id`` of ``self._ptr[i]``.
        """
        self._ptr[i].id = value
    @property
    def id(self):
        return self.get_id(0)
    @id.setter
    def id(self, int value):
        self.set_id(0,value)


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
    def get_location(self, i):
        """Get value of ``location`` of ``self._ptr[i]``.
        """
        return hipMemLocation.from_ptr(&self._ptr[i].location)
    @property
    def location(self):
        return self.get_location(0)
    def get_flags(self, i):
        """Get value of ``flags`` of ``self._ptr[i]``.
        """
        return hipMemAccessFlags(self._ptr[i].flags)
    def set_flags(self, i, value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemAccessFlags):
            raise TypeError("'value' must be of type 'hipMemAccessFlags'")
        self._ptr[i].flags = value.value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, value):
        self.set_flags(0,value)


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
    def get_allocType(self, i):
        """Get value of ``allocType`` of ``self._ptr[i]``.
        """
        return hipMemAllocationType(self._ptr[i].allocType)
    def set_allocType(self, i, value):
        """Set value ``allocType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemAllocationType):
            raise TypeError("'value' must be of type 'hipMemAllocationType'")
        self._ptr[i].allocType = value.value
    @property
    def allocType(self):
        return self.get_allocType(0)
    @allocType.setter
    def allocType(self, value):
        self.set_allocType(0,value)
    def get_handleTypes(self, i):
        """Get value of ``handleTypes`` of ``self._ptr[i]``.
        """
        return hipMemAllocationHandleType(self._ptr[i].handleTypes)
    def set_handleTypes(self, i, value):
        """Set value ``handleTypes`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemAllocationHandleType):
            raise TypeError("'value' must be of type 'hipMemAllocationHandleType'")
        self._ptr[i].handleTypes = value.value
    @property
    def handleTypes(self):
        return self.get_handleTypes(0)
    @handleTypes.setter
    def handleTypes(self, value):
        self.set_handleTypes(0,value)
    def get_location(self, i):
        """Get value of ``location`` of ``self._ptr[i]``.
        """
        return hipMemLocation.from_ptr(&self._ptr[i].location)
    @property
    def location(self):
        return self.get_location(0)
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



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
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters


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
    def get_x(self, i):
        """Get value ``x`` of ``self._ptr[i]``.
        """
        return self._ptr[i].x
    def set_x(self, i, uint32_t value):
        """Set value ``x`` of ``self._ptr[i]``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        return self.get_x(0)
    @x.setter
    def x(self, uint32_t value):
        self.set_x(0,value)
    def get_y(self, i):
        """Get value ``y`` of ``self._ptr[i]``.
        """
        return self._ptr[i].y
    def set_y(self, i, uint32_t value):
        """Set value ``y`` of ``self._ptr[i]``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        return self.get_y(0)
    @y.setter
    def y(self, uint32_t value):
        self.set_y(0,value)
    def get_z(self, i):
        """Get value ``z`` of ``self._ptr[i]``.
        """
        return self._ptr[i].z
    def set_z(self, i, uint32_t value):
        """Set value ``z`` of ``self._ptr[i]``.
        """
        self._ptr[i].z = value
    @property
    def z(self):
        return self.get_z(0)
    @z.setter
    def z(self, uint32_t value):
        self.set_z(0,value)



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
    def get_gridDim(self, i):
        """Get value of ``gridDim`` of ``self._ptr[i]``.
        """
        return dim3.from_ptr(&self._ptr[i].gridDim)
    @property
    def gridDim(self):
        return self.get_gridDim(0)
    def get_blockDim(self, i):
        """Get value of ``blockDim`` of ``self._ptr[i]``.
        """
        return dim3.from_ptr(&self._ptr[i].blockDim)
    @property
    def blockDim(self):
        return self.get_blockDim(0)
    def get_sharedMem(self, i):
        """Get value ``sharedMem`` of ``self._ptr[i]``.
        """
        return self._ptr[i].sharedMem
    def set_sharedMem(self, i, int value):
        """Set value ``sharedMem`` of ``self._ptr[i]``.
        """
        self._ptr[i].sharedMem = value
    @property
    def sharedMem(self):
        return self.get_sharedMem(0)
    @sharedMem.setter
    def sharedMem(self, int value):
        self.set_sharedMem(0,value)



cdef class hipLaunchParams:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipLaunchParams from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipLaunchParams`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipLaunchParams wrapper = hipLaunchParams.__new__(hipLaunchParams)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


class hipExternalMemoryHandleType_enum(enum.IntEnum):
    hipExternalMemoryHandleTypeOpaqueFd = chip.hipExternalMemoryHandleTypeOpaqueFd
    hipExternalMemoryHandleTypeOpaqueWin32 = chip.hipExternalMemoryHandleTypeOpaqueWin32
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = chip.hipExternalMemoryHandleTypeOpaqueWin32Kmt
    hipExternalMemoryHandleTypeD3D12Heap = chip.hipExternalMemoryHandleTypeD3D12Heap
    hipExternalMemoryHandleTypeD3D12Resource = chip.hipExternalMemoryHandleTypeD3D12Resource
    hipExternalMemoryHandleTypeD3D11Resource = chip.hipExternalMemoryHandleTypeD3D11Resource
    hipExternalMemoryHandleTypeD3D11ResourceKmt = chip.hipExternalMemoryHandleTypeD3D11ResourceKmt


cdef class hipExternalMemoryHandleType:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalMemoryHandleType from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipExternalMemoryHandleType`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalMemoryHandleType wrapper = hipExternalMemoryHandleType.__new__(hipExternalMemoryHandleType)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_fd(self, i):
        """Get value ``fd`` of ``self._ptr[i]``.
        """
        return self._ptr[i].fd
    def set_fd(self, i, int value):
        """Set value ``fd`` of ``self._ptr[i]``.
        """
        self._ptr[i].fd = value
    @property
    def fd(self):
        return self.get_fd(0)
    @fd.setter
    def fd(self, int value):
        self.set_fd(0,value)
    def get_win32(self, i):
        """Get value of ``win32`` of ``self._ptr[i]``.
        """
        return hipExternalMemoryHandleDesc_st_union_0_struct_0.from_ptr(&self._ptr[i].win32)
    @property
    def win32(self):
        return self.get_win32(0)



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
    def get_type(self, i):
        """Get value of ``type`` of ``self._ptr[i]``.
        """
        return hipExternalMemoryHandleType_enum(self._ptr[i].type)
    def set_type(self, i, value):
        """Set value ``type`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipExternalMemoryHandleType_enum):
            raise TypeError("'value' must be of type 'hipExternalMemoryHandleType_enum'")
        self._ptr[i].type = value.value
    @property
    def type(self):
        return self.get_type(0)
    @type.setter
    def type(self, value):
        self.set_type(0,value)
    def get_handle(self, i):
        """Get value of ``handle`` of ``self._ptr[i]``.
        """
        return hipExternalMemoryHandleDesc_st_union_0.from_ptr(&self._ptr[i].handle)
    @property
    def handle(self):
        return self.get_handle(0)
    def get_size(self, i):
        """Get value ``size`` of ``self._ptr[i]``.
        """
        return self._ptr[i].size
    def set_size(self, i, unsigned long long value):
        """Set value ``size`` of ``self._ptr[i]``.
        """
        self._ptr[i].size = value
    @property
    def size(self):
        return self.get_size(0)
    @size.setter
    def size(self, unsigned long long value):
        self.set_size(0,value)
    def get_flags(self, i):
        """Get value ``flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        self.set_flags(0,value)



cdef class hipExternalMemoryHandleDesc:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalMemoryHandleDesc from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipExternalMemoryHandleDesc`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalMemoryHandleDesc wrapper = hipExternalMemoryHandleDesc.__new__(hipExternalMemoryHandleDesc)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_offset(self, i):
        """Get value ``offset`` of ``self._ptr[i]``.
        """
        return self._ptr[i].offset
    def set_offset(self, i, unsigned long long value):
        """Set value ``offset`` of ``self._ptr[i]``.
        """
        self._ptr[i].offset = value
    @property
    def offset(self):
        return self.get_offset(0)
    @offset.setter
    def offset(self, unsigned long long value):
        self.set_offset(0,value)
    def get_size(self, i):
        """Get value ``size`` of ``self._ptr[i]``.
        """
        return self._ptr[i].size
    def set_size(self, i, unsigned long long value):
        """Set value ``size`` of ``self._ptr[i]``.
        """
        self._ptr[i].size = value
    @property
    def size(self):
        return self.get_size(0)
    @size.setter
    def size(self, unsigned long long value):
        self.set_size(0,value)
    def get_flags(self, i):
        """Get value ``flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        self.set_flags(0,value)



cdef class hipExternalMemoryBufferDesc:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalMemoryBufferDesc from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipExternalMemoryBufferDesc`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalMemoryBufferDesc wrapper = hipExternalMemoryBufferDesc.__new__(hipExternalMemoryBufferDesc)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class hipExternalMemory_t:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalMemory_t from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipExternalMemory_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalMemory_t wrapper = hipExternalMemory_t.__new__(hipExternalMemory_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


class hipExternalSemaphoreHandleType_enum(enum.IntEnum):
    hipExternalSemaphoreHandleTypeOpaqueFd = chip.hipExternalSemaphoreHandleTypeOpaqueFd
    hipExternalSemaphoreHandleTypeOpaqueWin32 = chip.hipExternalSemaphoreHandleTypeOpaqueWin32
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = chip.hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
    hipExternalSemaphoreHandleTypeD3D12Fence = chip.hipExternalSemaphoreHandleTypeD3D12Fence


cdef class hipExternalSemaphoreHandleType:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreHandleType from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreHandleType`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreHandleType wrapper = hipExternalSemaphoreHandleType.__new__(hipExternalSemaphoreHandleType)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_fd(self, i):
        """Get value ``fd`` of ``self._ptr[i]``.
        """
        return self._ptr[i].fd
    def set_fd(self, i, int value):
        """Set value ``fd`` of ``self._ptr[i]``.
        """
        self._ptr[i].fd = value
    @property
    def fd(self):
        return self.get_fd(0)
    @fd.setter
    def fd(self, int value):
        self.set_fd(0,value)
    def get_win32(self, i):
        """Get value of ``win32`` of ``self._ptr[i]``.
        """
        return hipExternalSemaphoreHandleDesc_st_union_0_struct_0.from_ptr(&self._ptr[i].win32)
    @property
    def win32(self):
        return self.get_win32(0)



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
    def get_type(self, i):
        """Get value of ``type`` of ``self._ptr[i]``.
        """
        return hipExternalSemaphoreHandleType_enum(self._ptr[i].type)
    def set_type(self, i, value):
        """Set value ``type`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipExternalSemaphoreHandleType_enum):
            raise TypeError("'value' must be of type 'hipExternalSemaphoreHandleType_enum'")
        self._ptr[i].type = value.value
    @property
    def type(self):
        return self.get_type(0)
    @type.setter
    def type(self, value):
        self.set_type(0,value)
    def get_handle(self, i):
        """Get value of ``handle`` of ``self._ptr[i]``.
        """
        return hipExternalSemaphoreHandleDesc_st_union_0.from_ptr(&self._ptr[i].handle)
    @property
    def handle(self):
        return self.get_handle(0)
    def get_flags(self, i):
        """Get value ``flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        self.set_flags(0,value)



cdef class hipExternalSemaphoreHandleDesc:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreHandleDesc from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreHandleDesc`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreHandleDesc wrapper = hipExternalSemaphoreHandleDesc.__new__(hipExternalSemaphoreHandleDesc)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class hipExternalSemaphore_t:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphore_t from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphore_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphore_t wrapper = hipExternalSemaphore_t.__new__(hipExternalSemaphore_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_value(self, i):
        """Get value ``value`` of ``self._ptr[i]``.
        """
        return self._ptr[i].value
    def set_value(self, i, unsigned long long value):
        """Set value ``value`` of ``self._ptr[i]``.
        """
        self._ptr[i].value = value
    @property
    def value(self):
        return self.get_value(0)
    @value.setter
    def value(self, unsigned long long value):
        self.set_value(0,value)



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
    def get_key(self, i):
        """Get value ``key`` of ``self._ptr[i]``.
        """
        return self._ptr[i].key
    def set_key(self, i, unsigned long long value):
        """Set value ``key`` of ``self._ptr[i]``.
        """
        self._ptr[i].key = value
    @property
    def key(self):
        return self.get_key(0)
    @key.setter
    def key(self, unsigned long long value):
        self.set_key(0,value)



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
    def get_fence(self, i):
        """Get value of ``fence`` of ``self._ptr[i]``.
        """
        return hipExternalSemaphoreSignalParams_st_struct_0_struct_0.from_ptr(&self._ptr[i].fence)
    @property
    def fence(self):
        return self.get_fence(0)
    def get_keyedMutex(self, i):
        """Get value of ``keyedMutex`` of ``self._ptr[i]``.
        """
        return hipExternalSemaphoreSignalParams_st_struct_0_struct_1.from_ptr(&self._ptr[i].keyedMutex)
    @property
    def keyedMutex(self):
        return self.get_keyedMutex(0)
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



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
    def get_params(self, i):
        """Get value of ``params`` of ``self._ptr[i]``.
        """
        return hipExternalSemaphoreSignalParams_st_struct_0.from_ptr(&self._ptr[i].params)
    @property
    def params(self):
        return self.get_params(0)
    def get_flags(self, i):
        """Get value ``flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        self.set_flags(0,value)
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



cdef class hipExternalSemaphoreSignalParams:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreSignalParams from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreSignalParams`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreSignalParams wrapper = hipExternalSemaphoreSignalParams.__new__(hipExternalSemaphoreSignalParams)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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
    def get_value(self, i):
        """Get value ``value`` of ``self._ptr[i]``.
        """
        return self._ptr[i].value
    def set_value(self, i, unsigned long long value):
        """Set value ``value`` of ``self._ptr[i]``.
        """
        self._ptr[i].value = value
    @property
    def value(self):
        return self.get_value(0)
    @value.setter
    def value(self, unsigned long long value):
        self.set_value(0,value)



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
    def get_key(self, i):
        """Get value ``key`` of ``self._ptr[i]``.
        """
        return self._ptr[i].key
    def set_key(self, i, unsigned long long value):
        """Set value ``key`` of ``self._ptr[i]``.
        """
        self._ptr[i].key = value
    @property
    def key(self):
        return self.get_key(0)
    @key.setter
    def key(self, unsigned long long value):
        self.set_key(0,value)
    def get_timeoutMs(self, i):
        """Get value ``timeoutMs`` of ``self._ptr[i]``.
        """
        return self._ptr[i].timeoutMs
    def set_timeoutMs(self, i, unsigned int value):
        """Set value ``timeoutMs`` of ``self._ptr[i]``.
        """
        self._ptr[i].timeoutMs = value
    @property
    def timeoutMs(self):
        return self.get_timeoutMs(0)
    @timeoutMs.setter
    def timeoutMs(self, unsigned int value):
        self.set_timeoutMs(0,value)



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
    def get_fence(self, i):
        """Get value of ``fence`` of ``self._ptr[i]``.
        """
        return hipExternalSemaphoreWaitParams_st_struct_0_struct_0.from_ptr(&self._ptr[i].fence)
    @property
    def fence(self):
        return self.get_fence(0)
    def get_keyedMutex(self, i):
        """Get value of ``keyedMutex`` of ``self._ptr[i]``.
        """
        return hipExternalSemaphoreWaitParams_st_struct_0_struct_1.from_ptr(&self._ptr[i].keyedMutex)
    @property
    def keyedMutex(self):
        return self.get_keyedMutex(0)
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



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
    def get_params(self, i):
        """Get value of ``params`` of ``self._ptr[i]``.
        """
        return hipExternalSemaphoreWaitParams_st_struct_0.from_ptr(&self._ptr[i].params)
    @property
    def params(self):
        return self.get_params(0)
    def get_flags(self, i):
        """Get value ``flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        self.set_flags(0,value)
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters



cdef class hipExternalSemaphoreWaitParams:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipExternalSemaphoreWaitParams from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipExternalSemaphoreWaitParams`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipExternalSemaphoreWaitParams wrapper = hipExternalSemaphoreWaitParams.__new__(hipExternalSemaphoreWaitParams)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


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



cdef class hipGraphicsResource:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipGraphicsResource from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipGraphicsResource`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipGraphicsResource wrapper = hipGraphicsResource.__new__(hipGraphicsResource)
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
    def get_blockDim(self, i):
        """Get value of ``blockDim`` of ``self._ptr[i]``.
        """
        return dim3.from_ptr(&self._ptr[i].blockDim)
    @property
    def blockDim(self):
        return self.get_blockDim(0)
    def get_gridDim(self, i):
        """Get value of ``gridDim`` of ``self._ptr[i]``.
        """
        return dim3.from_ptr(&self._ptr[i].gridDim)
    @property
    def gridDim(self):
        return self.get_gridDim(0)
    def get_sharedMemBytes(self, i):
        """Get value ``sharedMemBytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].sharedMemBytes
    def set_sharedMemBytes(self, i, unsigned int value):
        """Set value ``sharedMemBytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].sharedMemBytes = value
    @property
    def sharedMemBytes(self):
        return self.get_sharedMemBytes(0)
    @sharedMemBytes.setter
    def sharedMemBytes(self, unsigned int value):
        self.set_sharedMemBytes(0,value)



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
    def get_elementSize(self, i):
        """Get value ``elementSize`` of ``self._ptr[i]``.
        """
        return self._ptr[i].elementSize
    def set_elementSize(self, i, unsigned int value):
        """Set value ``elementSize`` of ``self._ptr[i]``.
        """
        self._ptr[i].elementSize = value
    @property
    def elementSize(self):
        return self.get_elementSize(0)
    @elementSize.setter
    def elementSize(self, unsigned int value):
        self.set_elementSize(0,value)
    def get_height(self, i):
        """Get value ``height`` of ``self._ptr[i]``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Set value ``height`` of ``self._ptr[i]``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        self.set_height(0,value)
    def get_pitch(self, i):
        """Get value ``pitch`` of ``self._ptr[i]``.
        """
        return self._ptr[i].pitch
    def set_pitch(self, i, int value):
        """Set value ``pitch`` of ``self._ptr[i]``.
        """
        self._ptr[i].pitch = value
    @property
    def pitch(self):
        return self.get_pitch(0)
    @pitch.setter
    def pitch(self, int value):
        self.set_pitch(0,value)
    def get_value(self, i):
        """Get value ``value`` of ``self._ptr[i]``.
        """
        return self._ptr[i].value
    def set_value(self, i, unsigned int value):
        """Set value ``value`` of ``self._ptr[i]``.
        """
        self._ptr[i].value = value
    @property
    def value(self):
        return self.get_value(0)
    @value.setter
    def value(self, unsigned int value):
        self.set_value(0,value)
    def get_width(self, i):
        """Get value ``width`` of ``self._ptr[i]``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Set value ``width`` of ``self._ptr[i]``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        self.set_width(0,value)


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
    def get_hitProp(self, i):
        """Get value of ``hitProp`` of ``self._ptr[i]``.
        """
        return hipAccessProperty(self._ptr[i].hitProp)
    def set_hitProp(self, i, value):
        """Set value ``hitProp`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipAccessProperty):
            raise TypeError("'value' must be of type 'hipAccessProperty'")
        self._ptr[i].hitProp = value.value
    @property
    def hitProp(self):
        return self.get_hitProp(0)
    @hitProp.setter
    def hitProp(self, value):
        self.set_hitProp(0,value)
    def get_hitRatio(self, i):
        """Get value ``hitRatio`` of ``self._ptr[i]``.
        """
        return self._ptr[i].hitRatio
    def set_hitRatio(self, i, float value):
        """Set value ``hitRatio`` of ``self._ptr[i]``.
        """
        self._ptr[i].hitRatio = value
    @property
    def hitRatio(self):
        return self.get_hitRatio(0)
    @hitRatio.setter
    def hitRatio(self, float value):
        self.set_hitRatio(0,value)
    def get_missProp(self, i):
        """Get value of ``missProp`` of ``self._ptr[i]``.
        """
        return hipAccessProperty(self._ptr[i].missProp)
    def set_missProp(self, i, value):
        """Set value ``missProp`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipAccessProperty):
            raise TypeError("'value' must be of type 'hipAccessProperty'")
        self._ptr[i].missProp = value.value
    @property
    def missProp(self):
        return self.get_missProp(0)
    @missProp.setter
    def missProp(self, value):
        self.set_missProp(0,value)
    def get_num_bytes(self, i):
        """Get value ``num_bytes`` of ``self._ptr[i]``.
        """
        return self._ptr[i].num_bytes
    def set_num_bytes(self, i, int value):
        """Set value ``num_bytes`` of ``self._ptr[i]``.
        """
        self._ptr[i].num_bytes = value
    @property
    def num_bytes(self):
        return self.get_num_bytes(0)
    @num_bytes.setter
    def num_bytes(self, int value):
        self.set_num_bytes(0,value)



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
    def get_accessPolicyWindow(self, i):
        """Get value of ``accessPolicyWindow`` of ``self._ptr[i]``.
        """
        return hipAccessPolicyWindow.from_ptr(&self._ptr[i].accessPolicyWindow)
    @property
    def accessPolicyWindow(self):
        return self.get_accessPolicyWindow(0)
    def get_cooperative(self, i):
        """Get value ``cooperative`` of ``self._ptr[i]``.
        """
        return self._ptr[i].cooperative
    def set_cooperative(self, i, int value):
        """Set value ``cooperative`` of ``self._ptr[i]``.
        """
        self._ptr[i].cooperative = value
    @property
    def cooperative(self):
        return self.get_cooperative(0)
    @cooperative.setter
    def cooperative(self, int value):
        self.set_cooperative(0,value)


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
    def get_compressionType(self, i):
        """Get value ``compressionType`` of ``self._ptr[i]``.
        """
        return self._ptr[i].compressionType
    def set_compressionType(self, i, unsigned char value):
        """Set value ``compressionType`` of ``self._ptr[i]``.
        """
        self._ptr[i].compressionType = value
    @property
    def compressionType(self):
        return self.get_compressionType(0)
    @compressionType.setter
    def compressionType(self, unsigned char value):
        self.set_compressionType(0,value)
    def get_gpuDirectRDMACapable(self, i):
        """Get value ``gpuDirectRDMACapable`` of ``self._ptr[i]``.
        """
        return self._ptr[i].gpuDirectRDMACapable
    def set_gpuDirectRDMACapable(self, i, unsigned char value):
        """Set value ``gpuDirectRDMACapable`` of ``self._ptr[i]``.
        """
        self._ptr[i].gpuDirectRDMACapable = value
    @property
    def gpuDirectRDMACapable(self):
        return self.get_gpuDirectRDMACapable(0)
    @gpuDirectRDMACapable.setter
    def gpuDirectRDMACapable(self, unsigned char value):
        self.set_gpuDirectRDMACapable(0,value)
    def get_usage(self, i):
        """Get value ``usage`` of ``self._ptr[i]``.
        """
        return self._ptr[i].usage
    def set_usage(self, i, unsigned short value):
        """Set value ``usage`` of ``self._ptr[i]``.
        """
        self._ptr[i].usage = value
    @property
    def usage(self):
        return self.get_usage(0)
    @usage.setter
    def usage(self, unsigned short value):
        self.set_usage(0,value)



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
    def get_type(self, i):
        """Get value of ``type`` of ``self._ptr[i]``.
        """
        return hipMemAllocationType(self._ptr[i].type)
    def set_type(self, i, value):
        """Set value ``type`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemAllocationType):
            raise TypeError("'value' must be of type 'hipMemAllocationType'")
        self._ptr[i].type = value.value
    @property
    def type(self):
        return self.get_type(0)
    @type.setter
    def type(self, value):
        self.set_type(0,value)
    def get_requestedHandleType(self, i):
        """Get value of ``requestedHandleType`` of ``self._ptr[i]``.
        """
        return hipMemAllocationHandleType(self._ptr[i].requestedHandleType)
    def set_requestedHandleType(self, i, value):
        """Set value ``requestedHandleType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemAllocationHandleType):
            raise TypeError("'value' must be of type 'hipMemAllocationHandleType'")
        self._ptr[i].requestedHandleType = value.value
    @property
    def requestedHandleType(self):
        return self.get_requestedHandleType(0)
    @requestedHandleType.setter
    def requestedHandleType(self, value):
        self.set_requestedHandleType(0,value)
    def get_location(self, i):
        """Get value of ``location`` of ``self._ptr[i]``.
        """
        return hipMemLocation.from_ptr(&self._ptr[i].location)
    @property
    def location(self):
        return self.get_location(0)
    def get_allocFlags(self, i):
        """Get value of ``allocFlags`` of ``self._ptr[i]``.
        """
        return hipMemAllocationProp_struct_0.from_ptr(&self._ptr[i].allocFlags)
    @property
    def allocFlags(self):
        return self.get_allocFlags(0)



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
    def get_mipmap(self, i):
        """Get value of ``mipmap`` of ``self._ptr[i]``.
        """
        return hipMipmappedArray.from_ptr(&self._ptr[i].mipmap)
    @property
    def mipmap(self):
        return self.get_mipmap(0)



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
    def get_level(self, i):
        """Get value ``level`` of ``self._ptr[i]``.
        """
        return self._ptr[i].level
    def set_level(self, i, unsigned int value):
        """Set value ``level`` of ``self._ptr[i]``.
        """
        self._ptr[i].level = value
    @property
    def level(self):
        return self.get_level(0)
    @level.setter
    def level(self, unsigned int value):
        self.set_level(0,value)
    def get_layer(self, i):
        """Get value ``layer`` of ``self._ptr[i]``.
        """
        return self._ptr[i].layer
    def set_layer(self, i, unsigned int value):
        """Set value ``layer`` of ``self._ptr[i]``.
        """
        self._ptr[i].layer = value
    @property
    def layer(self):
        return self.get_layer(0)
    @layer.setter
    def layer(self, unsigned int value):
        self.set_layer(0,value)
    def get_offsetX(self, i):
        """Get value ``offsetX`` of ``self._ptr[i]``.
        """
        return self._ptr[i].offsetX
    def set_offsetX(self, i, unsigned int value):
        """Set value ``offsetX`` of ``self._ptr[i]``.
        """
        self._ptr[i].offsetX = value
    @property
    def offsetX(self):
        return self.get_offsetX(0)
    @offsetX.setter
    def offsetX(self, unsigned int value):
        self.set_offsetX(0,value)
    def get_offsetY(self, i):
        """Get value ``offsetY`` of ``self._ptr[i]``.
        """
        return self._ptr[i].offsetY
    def set_offsetY(self, i, unsigned int value):
        """Set value ``offsetY`` of ``self._ptr[i]``.
        """
        self._ptr[i].offsetY = value
    @property
    def offsetY(self):
        return self.get_offsetY(0)
    @offsetY.setter
    def offsetY(self, unsigned int value):
        self.set_offsetY(0,value)
    def get_offsetZ(self, i):
        """Get value ``offsetZ`` of ``self._ptr[i]``.
        """
        return self._ptr[i].offsetZ
    def set_offsetZ(self, i, unsigned int value):
        """Set value ``offsetZ`` of ``self._ptr[i]``.
        """
        self._ptr[i].offsetZ = value
    @property
    def offsetZ(self):
        return self.get_offsetZ(0)
    @offsetZ.setter
    def offsetZ(self, unsigned int value):
        self.set_offsetZ(0,value)
    def get_extentWidth(self, i):
        """Get value ``extentWidth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].extentWidth
    def set_extentWidth(self, i, unsigned int value):
        """Set value ``extentWidth`` of ``self._ptr[i]``.
        """
        self._ptr[i].extentWidth = value
    @property
    def extentWidth(self):
        return self.get_extentWidth(0)
    @extentWidth.setter
    def extentWidth(self, unsigned int value):
        self.set_extentWidth(0,value)
    def get_extentHeight(self, i):
        """Get value ``extentHeight`` of ``self._ptr[i]``.
        """
        return self._ptr[i].extentHeight
    def set_extentHeight(self, i, unsigned int value):
        """Set value ``extentHeight`` of ``self._ptr[i]``.
        """
        self._ptr[i].extentHeight = value
    @property
    def extentHeight(self):
        return self.get_extentHeight(0)
    @extentHeight.setter
    def extentHeight(self, unsigned int value):
        self.set_extentHeight(0,value)
    def get_extentDepth(self, i):
        """Get value ``extentDepth`` of ``self._ptr[i]``.
        """
        return self._ptr[i].extentDepth
    def set_extentDepth(self, i, unsigned int value):
        """Set value ``extentDepth`` of ``self._ptr[i]``.
        """
        self._ptr[i].extentDepth = value
    @property
    def extentDepth(self):
        return self.get_extentDepth(0)
    @extentDepth.setter
    def extentDepth(self, unsigned int value):
        self.set_extentDepth(0,value)



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
    def get_layer(self, i):
        """Get value ``layer`` of ``self._ptr[i]``.
        """
        return self._ptr[i].layer
    def set_layer(self, i, unsigned int value):
        """Set value ``layer`` of ``self._ptr[i]``.
        """
        self._ptr[i].layer = value
    @property
    def layer(self):
        return self.get_layer(0)
    @layer.setter
    def layer(self, unsigned int value):
        self.set_layer(0,value)
    def get_offset(self, i):
        """Get value ``offset`` of ``self._ptr[i]``.
        """
        return self._ptr[i].offset
    def set_offset(self, i, unsigned long long value):
        """Set value ``offset`` of ``self._ptr[i]``.
        """
        self._ptr[i].offset = value
    @property
    def offset(self):
        return self.get_offset(0)
    @offset.setter
    def offset(self, unsigned long long value):
        self.set_offset(0,value)
    def get_size(self, i):
        """Get value ``size`` of ``self._ptr[i]``.
        """
        return self._ptr[i].size
    def set_size(self, i, unsigned long long value):
        """Set value ``size`` of ``self._ptr[i]``.
        """
        self._ptr[i].size = value
    @property
    def size(self):
        return self.get_size(0)
    @size.setter
    def size(self, unsigned long long value):
        self.set_size(0,value)



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
    def get_sparseLevel(self, i):
        """Get value of ``sparseLevel`` of ``self._ptr[i]``.
        """
        return hipArrayMapInfo_union_1_struct_0.from_ptr(&self._ptr[i].sparseLevel)
    @property
    def sparseLevel(self):
        return self.get_sparseLevel(0)
    def get_miptail(self, i):
        """Get value of ``miptail`` of ``self._ptr[i]``.
        """
        return hipArrayMapInfo_union_1_struct_1.from_ptr(&self._ptr[i].miptail)
    @property
    def miptail(self):
        return self.get_miptail(0)



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
    def get_resourceType(self, i):
        """Get value of ``resourceType`` of ``self._ptr[i]``.
        """
        return hipResourceType(self._ptr[i].resourceType)
    def set_resourceType(self, i, value):
        """Set value ``resourceType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipResourceType):
            raise TypeError("'value' must be of type 'hipResourceType'")
        self._ptr[i].resourceType = value.value
    @property
    def resourceType(self):
        return self.get_resourceType(0)
    @resourceType.setter
    def resourceType(self, value):
        self.set_resourceType(0,value)
    def get_resource(self, i):
        """Get value of ``resource`` of ``self._ptr[i]``.
        """
        return hipArrayMapInfo_union_0.from_ptr(&self._ptr[i].resource)
    @property
    def resource(self):
        return self.get_resource(0)
    def get_subresourceType(self, i):
        """Get value of ``subresourceType`` of ``self._ptr[i]``.
        """
        return hipArraySparseSubresourceType(self._ptr[i].subresourceType)
    def set_subresourceType(self, i, value):
        """Set value ``subresourceType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipArraySparseSubresourceType):
            raise TypeError("'value' must be of type 'hipArraySparseSubresourceType'")
        self._ptr[i].subresourceType = value.value
    @property
    def subresourceType(self):
        return self.get_subresourceType(0)
    @subresourceType.setter
    def subresourceType(self, value):
        self.set_subresourceType(0,value)
    def get_subresource(self, i):
        """Get value of ``subresource`` of ``self._ptr[i]``.
        """
        return hipArrayMapInfo_union_1.from_ptr(&self._ptr[i].subresource)
    @property
    def subresource(self):
        return self.get_subresource(0)
    def get_memOperationType(self, i):
        """Get value of ``memOperationType`` of ``self._ptr[i]``.
        """
        return hipMemOperationType(self._ptr[i].memOperationType)
    def set_memOperationType(self, i, value):
        """Set value ``memOperationType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemOperationType):
            raise TypeError("'value' must be of type 'hipMemOperationType'")
        self._ptr[i].memOperationType = value.value
    @property
    def memOperationType(self):
        return self.get_memOperationType(0)
    @memOperationType.setter
    def memOperationType(self, value):
        self.set_memOperationType(0,value)
    def get_memHandleType(self, i):
        """Get value of ``memHandleType`` of ``self._ptr[i]``.
        """
        return hipMemHandleType(self._ptr[i].memHandleType)
    def set_memHandleType(self, i, value):
        """Set value ``memHandleType`` of ``self._ptr[i]``.
        """
        if not isinstance(value, hipMemHandleType):
            raise TypeError("'value' must be of type 'hipMemHandleType'")
        self._ptr[i].memHandleType = value.value
    @property
    def memHandleType(self):
        return self.get_memHandleType(0)
    @memHandleType.setter
    def memHandleType(self, value):
        self.set_memHandleType(0,value)
    def get_memHandle(self, i):
        """Get value of ``memHandle`` of ``self._ptr[i]``.
        """
        return hipArrayMapInfo_union_2.from_ptr(&self._ptr[i].memHandle)
    @property
    def memHandle(self):
        return self.get_memHandle(0)
    def get_offset(self, i):
        """Get value ``offset`` of ``self._ptr[i]``.
        """
        return self._ptr[i].offset
    def set_offset(self, i, unsigned long long value):
        """Set value ``offset`` of ``self._ptr[i]``.
        """
        self._ptr[i].offset = value
    @property
    def offset(self):
        return self.get_offset(0)
    @offset.setter
    def offset(self, unsigned long long value):
        self.set_offset(0,value)
    def get_deviceBitMask(self, i):
        """Get value ``deviceBitMask`` of ``self._ptr[i]``.
        """
        return self._ptr[i].deviceBitMask
    def set_deviceBitMask(self, i, unsigned int value):
        """Set value ``deviceBitMask`` of ``self._ptr[i]``.
        """
        self._ptr[i].deviceBitMask = value
    @property
    def deviceBitMask(self):
        return self.get_deviceBitMask(0)
    @deviceBitMask.setter
    def deviceBitMask(self, unsigned int value):
        self.set_deviceBitMask(0,value)
    def get_flags(self, i):
        """Get value ``flags`` of ``self._ptr[i]``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Set value ``flags`` of ``self._ptr[i]``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        self.set_flags(0,value)
    def get_reserved(self, i):
        """Get value of ``reserved`` of ``self._ptr[i]``.
        """
        return self._ptr[i].reserved
    @property
    def reserved(self):
        return self.get_reserved(0)
    # TODO is_basic_type_constantarray: add setters


def hipInit(unsigned int flags):
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
    hipInit_____retval = hipError_t(chip.hipInit(flags))    # fully specified
    return hipInit_____retval


def hipDriverGetVersion():
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
    cdef int driverVersion
    hipDriverGetVersion_____retval = hipError_t(chip.hipDriverGetVersion(&driverVersion))    # fully specified
    return (hipDriverGetVersion_____retval,driverVersion)


def hipRuntimeGetVersion():
    """@brief Returns the approximate HIP Runtime version.
    @param [out] runtimeVersion
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning The version definition of HIP runtime is different from CUDA.
    On AMD platform, the function returns HIP runtime version,
    while on NVIDIA platform, it returns CUDA runtime version.
    And there is no mapping/correlation between HIP version and CUDA version.
    @see hipDriverGetVersion
    """
    cdef int runtimeVersion
    hipRuntimeGetVersion_____retval = hipError_t(chip.hipRuntimeGetVersion(&runtimeVersion))    # fully specified
    return (hipRuntimeGetVersion_____retval,runtimeVersion)


def hipDeviceGet(int ordinal):
    """@brief Returns a handle to a compute device
    @param [out] device
    @param [in] ordinal
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    cdef int device
    hipDeviceGet_____retval = hipError_t(chip.hipDeviceGet(&device,ordinal))    # fully specified
    return (hipDeviceGet_____retval,device)


def hipDeviceComputeCapability(hipDevice_t device):
    """@brief Returns the compute capability of the device
    @param [out] major
    @param [out] minor
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    cdef int major
    cdef int minor
    hipDeviceComputeCapability_____retval = hipError_t(chip.hipDeviceComputeCapability(&major,&minor,device))    # fully specified
    return (hipDeviceComputeCapability_____retval,major,minor)


def hipDeviceGetName(char * name, int len, hipDevice_t device):
    """@brief Returns an identifer string for the device.
    @param [out] name
    @param [in] len
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    hipDeviceGetName_____retval = hipError_t(chip.hipDeviceGetName(name,len,device))    # fully specified
    return hipDeviceGetName_____retval


def hipDeviceGetUuid(uuid, hipDevice_t device):
    """@brief Returns an UUID for the device.[BETA]
    @param [out] uuid
    @param [in] device
    @beta This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotInitialized,
    #hipErrorDeinitialized
    """
    pass

def hipDeviceGetP2PAttribute(attr, int srcDevice, int dstDevice):
    """@brief Returns a value for attr of link between two devices
    @param [out] value
    @param [in] attr
    @param [in] srcDevice
    @param [in] dstDevice
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    cdef int value
    pass

def hipDeviceGetPCIBusId(char * pciBusId, int len, int device):
    """@brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
    @param [out] pciBusId
    @param [in] len
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    hipDeviceGetPCIBusId_____retval = hipError_t(chip.hipDeviceGetPCIBusId(pciBusId,len,device))    # fully specified
    return hipDeviceGetPCIBusId_____retval


def hipDeviceGetByPCIBusId(const char * pciBusId):
    """@brief Returns a handle to a compute device.
    @param [out] device handle
    @param [in] PCI Bus ID
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    """
    cdef int device
    hipDeviceGetByPCIBusId_____retval = hipError_t(chip.hipDeviceGetByPCIBusId(&device,pciBusId))    # fully specified
    return (hipDeviceGetByPCIBusId_____retval,device)


def hipDeviceTotalMem(hipDevice_t device):
    """@brief Returns the total amount of memory on the device.
    @param [out] bytes
    @param [in] device
    @returns #hipSuccess, #hipErrorInvalidDevice
    """
    cdef int bytes
    hipDeviceTotalMem_____retval = hipError_t(chip.hipDeviceTotalMem(&bytes,device))    # fully specified
    return (hipDeviceTotalMem_____retval,bytes)


def hipDeviceSynchronize():
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
    hipDeviceSynchronize_____retval = hipError_t(chip.hipDeviceSynchronize())    # fully specified
    return hipDeviceSynchronize_____retval


def hipDeviceReset():
    """@brief The state of current device is discarded and updated to a fresh state.
    Calling this function deletes all streams created, memory allocated, kernels running, events
    created. Make sure that no other thread is using the device or streams, memory, kernels, events
    associated with the current device.
    @returns #hipSuccess
    @see hipDeviceSynchronize
    """
    hipDeviceReset_____retval = hipError_t(chip.hipDeviceReset())    # fully specified
    return hipDeviceReset_____retval


def hipSetDevice(int deviceId):
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
    hipSetDevice_____retval = hipError_t(chip.hipSetDevice(deviceId))    # fully specified
    return hipSetDevice_____retval


def hipGetDevice():
    """@brief Return the default device id for the calling host thread.
    @param [out] device *device is written with the default device
    HIP maintains an default device for each thread using thread-local-storage.
    This device is used implicitly for HIP runtime APIs called by this thread.
    hipGetDevice returns in * @p device the default device for the calling host thread.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see hipSetDevice, hipGetDevicesizeBytes
    """
    cdef int deviceId
    hipGetDevice_____retval = hipError_t(chip.hipGetDevice(&deviceId))    # fully specified
    return (hipGetDevice_____retval,deviceId)


def hipGetDeviceCount():
    """@brief Return number of compute-capable devices.
    @param [output] count Returns number of compute-capable devices.
    @returns #hipSuccess, #hipErrorNoDevice
    Returns in @p *count the number of devices that have ability to run compute commands.  If there
    are no such devices, then @ref hipGetDeviceCount will return #hipErrorNoDevice. If 1 or more
    devices can be found, then hipGetDeviceCount returns #hipSuccess.
    """
    cdef int count
    hipGetDeviceCount_____retval = hipError_t(chip.hipGetDeviceCount(&count))    # fully specified
    return (hipGetDeviceCount_____retval,count)


def hipDeviceGetAttribute(attr, int deviceId):
    """@brief Query for a specific device attribute.
    @param [out] pi pointer to value to return
    @param [in] attr attribute to query
    @param [in] deviceId which device to query for information
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    """
    cdef int pi
    pass

def hipDeviceGetDefaultMemPool(int device):
    """@brief Returns the default memory pool of the specified device
    @param [out] mem_pool Default memory pool to return
    @param [in] device    Device index for query the default memory pool
    @returns #chipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotSupported
    @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
    hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    mem_pool = ihipMemPoolHandle_t.from_ptr(NULL,owner=True)
    hipDeviceGetDefaultMemPool_____retval = hipError_t(chip.hipDeviceGetDefaultMemPool(&mem_pool._ptr,device))    # fully specified
    return (hipDeviceGetDefaultMemPool_____retval,mem_pool)


def hipDeviceSetMemPool(int device, mem_pool):
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

def hipDeviceGetMemPool(int device):
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
    mem_pool = ihipMemPoolHandle_t.from_ptr(NULL,owner=True)
    hipDeviceGetMemPool_____retval = hipError_t(chip.hipDeviceGetMemPool(&mem_pool._ptr,device))    # fully specified
    return (hipDeviceGetMemPool_____retval,mem_pool)


def hipGetDeviceProperties(prop, int deviceId):
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

def hipDeviceSetCacheConfig(cacheConfig):
    """@brief Set L1/Shared cache partition.
    @param [in] cacheConfig
    @returns #hipSuccess, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    on those architectures.
    """
    pass

def hipDeviceGetCacheConfig():
    """@brief Get Cache configuration for a specific Device
    @param [out] cacheConfig
    @returns #hipSuccess, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    on those architectures.
    """
    cdef chip.hipFuncCache_t cacheConfig
    hipDeviceGetCacheConfig_____retval = hipError_t(chip.hipDeviceGetCacheConfig(&cacheConfig))    # fully specified
    return (hipDeviceGetCacheConfig_____retval,hipFuncCache_t(cacheConfig))


def hipDeviceGetLimit(limit):
    """@brief Get Resource limits of current device
    @param [out] pValue
    @param [in]  limit
    @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
    Note: Currently, only hipLimitMallocHeapSize is available
    """
    cdef int pValue
    pass

def hipDeviceSetLimit(limit, int value):
    """@brief Set Resource limits of current device
    @param [in] limit
    @param [in] value
    @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
    """
    pass

def hipDeviceGetSharedMemConfig():
    """@brief Returns bank width of shared memory for current device
    @param [out] pConfig
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    cdef chip.hipSharedMemConfig pConfig
    hipDeviceGetSharedMemConfig_____retval = hipError_t(chip.hipDeviceGetSharedMemConfig(&pConfig))    # fully specified
    return (hipDeviceGetSharedMemConfig_____retval,hipSharedMemConfig(pConfig))


def hipGetDeviceFlags():
    """@brief Gets the flags set for current device
    @param [out] flags
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    """
    cdef unsigned int flags
    hipGetDeviceFlags_____retval = hipError_t(chip.hipGetDeviceFlags(&flags))    # fully specified
    return (hipGetDeviceFlags_____retval,flags)


def hipDeviceSetSharedMemConfig(config):
    """@brief The bank width of shared memory on current device is set
    @param [in] config
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipSetDeviceFlags(unsigned int flags):
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
    hipSetDeviceFlags_____retval = hipError_t(chip.hipSetDeviceFlags(flags))    # fully specified
    return hipSetDeviceFlags_____retval


def hipChooseDevice(prop):
    """@brief Device which matches hipDeviceProp_t is returned
    @param [out] device ID
    @param [in]  device properties pointer
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    cdef int device
    pass

def hipExtGetLinkTypeAndHopCount(int device1, int device2):
    """@brief Returns the link type and hop count between two devices
    @param [in] device1 Ordinal for device1
    @param [in] device2 Ordinal for device2
    @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
    @param [out] hopcount Returns the hop count between the two devices
    Queries and returns the HSA link type and the hop count between the two specified devices.
    @returns #hipSuccess, #hipInvalidDevice, #hipErrorRuntimeOther
    """
    cdef unsigned int linktype
    cdef unsigned int hopcount
    hipExtGetLinkTypeAndHopCount_____retval = hipError_t(chip.hipExtGetLinkTypeAndHopCount(device1,device2,&linktype,&hopcount))    # fully specified
    return (hipExtGetLinkTypeAndHopCount_____retval,linktype,hopcount)


def hipIpcGetMemHandle(handle, devPtr):
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

def hipIpcOpenMemHandle(handle, unsigned int flags):
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

def hipIpcCloseMemHandle(devPtr):
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

def hipIpcGetEventHandle(handle, event):
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

def hipIpcOpenEventHandle(handle):
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
    event = ihipEvent_t.from_ptr(NULL,owner=True)
    pass

def hipFuncSetAttribute(func, attr, int value):
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

def hipFuncSetCacheConfig(func, config):
    """@brief Set Cache configuration for a specific function
    @param [in] config;
    @returns #hipSuccess, #hipErrorNotInitialized
    Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
    on those architectures.
    """
    pass

def hipFuncSetSharedMemConfig(func, config):
    """@brief Set shared memory configuation for a specific function
    @param [in] func
    @param [in] config
    @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
    Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    """
    pass

def hipGetLastError():
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
    hipGetLastError_____retval = hipError_t(chip.hipGetLastError())    # fully specified
    return hipGetLastError_____retval


def hipPeekAtLastError():
    """@brief Return last error returned by any HIP runtime API call.
    @return #hipSuccess
    Returns the last error that has been returned by any of the runtime calls in the same host
    thread. Unlike hipGetLastError, this function does not reset the saved error code.
    @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    hipPeekAtLastError_____retval = hipError_t(chip.hipPeekAtLastError())    # fully specified
    return hipPeekAtLastError_____retval


def hipGetErrorName(hip_error):
    """@brief Return hip error as text string form.
    @param hip_error Error code to convert to name.
    @return const char pointer to the NULL-terminated error name
    @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipGetErrorString(hipError):
    """@brief Return handy text string message to explain the error which occurred
    @param hipError Error code to convert to string.
    @return const char pointer to the NULL-terminated error string
    @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipDrvGetErrorName(hipError):
    """@brief Return hip error as text string form.
    @param [in] hipError Error code to convert to string.
    @param [out] const char pointer to the NULL-terminated error string
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipDrvGetErrorString(hipError):
    """@brief Return handy text string message to explain the error which occurred
    @param [in] hipError Error code to convert to string.
    @param [out] const char pointer to the NULL-terminated error string
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
    """
    pass

def hipStreamCreate():
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
    stream = ihipStream_t.from_ptr(NULL,owner=True)
    hipStreamCreate_____retval = hipError_t(chip.hipStreamCreate(&stream._ptr))    # fully specified
    return (hipStreamCreate_____retval,stream)


def hipStreamCreateWithFlags(unsigned int flags):
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
    stream = ihipStream_t.from_ptr(NULL,owner=True)
    hipStreamCreateWithFlags_____retval = hipError_t(chip.hipStreamCreateWithFlags(&stream._ptr,flags))    # fully specified
    return (hipStreamCreateWithFlags_____retval,stream)


def hipStreamCreateWithPriority(unsigned int flags, int priority):
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
    stream = ihipStream_t.from_ptr(NULL,owner=True)
    hipStreamCreateWithPriority_____retval = hipError_t(chip.hipStreamCreateWithPriority(&stream._ptr,flags,priority))    # fully specified
    return (hipStreamCreateWithPriority_____retval,stream)


def hipDeviceGetStreamPriorityRange():
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
    cdef int leastPriority
    cdef int greatestPriority
    hipDeviceGetStreamPriorityRange_____retval = hipError_t(chip.hipDeviceGetStreamPriorityRange(&leastPriority,&greatestPriority))    # fully specified
    return (hipDeviceGetStreamPriorityRange_____retval,leastPriority,greatestPriority)


def hipStreamDestroy(stream):
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

def hipStreamQuery(stream):
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

def hipStreamSynchronize(stream):
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

def hipStreamWaitEvent(stream, event, unsigned int flags):
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

def hipStreamGetFlags(stream):
    """@brief Return flags associated with this stream.
    @param[in] stream stream to be queried
    @param[in,out] flags Pointer to an unsigned integer in which the stream's flags are returned
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
    @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
    Return flags associated with this stream in *@p flags.
    @see hipStreamCreateWithFlags
    """
    cdef unsigned int flags
    pass

def hipStreamGetPriority(stream):
    """@brief Query the priority of a stream.
    @param[in] stream stream to be queried
    @param[in,out] priority Pointer to an unsigned integer in which the stream's priority is returned
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
    @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
    Query the priority of a stream. The priority is returned in in priority.
    @see hipStreamCreateWithFlags
    """
    cdef int priority
    pass

def hipExtStreamCreateWithCUMask(uint32_t cuMaskSize):
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
    stream = ihipStream_t.from_ptr(NULL,owner=True)
    cdef const unsigned int cuMask
    hipExtStreamCreateWithCUMask_____retval = hipError_t(chip.hipExtStreamCreateWithCUMask(&stream._ptr,cuMaskSize,&cuMask))    # fully specified
    return (hipExtStreamCreateWithCUMask_____retval,stream,cuMask)


def hipExtStreamGetCUMask(stream, uint32_t cuMaskSize):
    """@brief Get CU mask associated with an asynchronous stream
    @param[in] stream stream to be queried
    @param[in] cuMaskSize number of the block of memories (uint32_t *) allocated by user
    @param[out] cuMask Pointer to a pre-allocated block of memories (uint32_t *) in which
    the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
    each active bit represents one active CU
    @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
    @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
    """
    cdef unsigned int cuMask
    pass


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


def hipStreamAddCallback(stream, callback, userData, unsigned int flags):
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

def hipStreamWaitValue32(stream, ptr, uint32_t value, unsigned int flags, uint32_t mask):
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

def hipStreamWaitValue64(stream, ptr, uint64_t value, unsigned int flags, uint64_t mask):
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

def hipStreamWriteValue32(stream, ptr, uint32_t value, unsigned int flags):
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

def hipStreamWriteValue64(stream, ptr, uint64_t value, unsigned int flags):
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

def hipEventCreateWithFlags(unsigned int flags):
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
    event = ihipEvent_t.from_ptr(NULL,owner=True)
    hipEventCreateWithFlags_____retval = hipError_t(chip.hipEventCreateWithFlags(&event._ptr,flags))    # fully specified
    return (hipEventCreateWithFlags_____retval,event)


def hipEventCreate():
    """Create an event
    @param[in,out] event Returns the newly created event.
    @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
    #hipErrorLaunchFailure, #hipErrorOutOfMemory
    @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize,
    hipEventDestroy, hipEventElapsedTime
    """
    event = ihipEvent_t.from_ptr(NULL,owner=True)
    hipEventCreate_____retval = hipError_t(chip.hipEventCreate(&event._ptr))    # fully specified
    return (hipEventCreate_____retval,event)


def hipEventRecord(event, stream):
    """
    """
    pass

def hipEventDestroy(event):
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

def hipEventSynchronize(event):
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

def hipEventElapsedTime(start, stop):
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
    cdef float ms
    pass

def hipEventQuery(event):
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

def hipPointerGetAttributes(attributes, ptr):
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

def hipPointerGetAttribute(data, attribute, ptr):
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

def hipDrvPointerGetAttributes(unsigned int numAttributes, ptr):
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
    cdef chip.hipPointer_attribute attributes
    pass

def hipImportExternalSemaphore(semHandleDesc):
    """@brief Imports an external semaphore.
    @param[out] extSem_out  External semaphores to be waited on
    @param[in] semHandleDesc Semaphore import handle descriptor
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipSignalExternalSemaphoresAsync(paramsArray, unsigned int numExtSems, stream):
    """@brief Signals a set of external semaphore objects.
    @param[in] extSem_out  External semaphores to be waited on
    @param[in] paramsArray Array of semaphore parameters
    @param[in] numExtSems Number of semaphores to wait on
    @param[in] stream Stream to enqueue the wait operations in
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipWaitExternalSemaphoresAsync(paramsArray, unsigned int numExtSems, stream):
    """@brief Waits on a set of external semaphore objects
    @param[in] extSem_out  External semaphores to be waited on
    @param[in] paramsArray Array of semaphore parameters
    @param[in] numExtSems Number of semaphores to wait on
    @param[in] stream Stream to enqueue the wait operations in
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipDestroyExternalSemaphore(extSem):
    """@brief Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.
    @param[in] extSem handle to an external memory object
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipImportExternalMemory(memHandleDesc):
    """@brief Imports an external memory object.
    @param[out] extMem_out  Returned handle to an external memory object
    @param[in]  memHandleDesc Memory import handle descriptor
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipExternalMemoryGetMappedBuffer(extMem, bufferDesc):
    """@brief Maps a buffer onto an imported memory object.
    @param[out] devPtr Returned device pointer to buffer
    @param[in]  extMem  Handle to external memory object
    @param[in]  bufferDesc  Buffer descriptor
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipDestroyExternalMemory(extMem):
    """@brief Destroys an external memory object.
    @param[in] extMem  External memory object to be destroyed
    @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @see
    """
    pass

def hipMalloc(int size):
    """@brief Allocate memory on the default accelerator
    @param[out] ptr Pointer to the allocated memory
    @param[in]  size Requested memory size
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
    @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
    hipHostFree, hipHostMalloc
    """
    pass

def hipExtMallocWithFlags(int sizeBytes, unsigned int flags):
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

def hipMallocHost(int size):
    """@brief Allocate pinned host memory [Deprecated]
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @deprecated use hipHostMalloc() instead
    """
    pass

def hipMemAllocHost(int size):
    """@brief Allocate pinned host memory [Deprecated]
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @deprecated use hipHostMalloc() instead
    """
    pass

def hipHostMalloc(int size, unsigned int flags):
    """@brief Allocate device accessible page locked host memory
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    @param[in]  flags Type of host memory allocation
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @see hipSetDeviceFlags, hipHostFree
    """
    pass

def hipMallocManaged(int size, unsigned int flags):
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

def hipMemPrefetchAsync(dev_ptr, int count, int device, stream):
    """@brief Prefetches memory to the specified destination device using HIP.
    @param [in] dev_ptr  pointer to be prefetched
    @param [in] count    size in bytes for prefetching
    @param [in] device   destination device to prefetch to
    @param [in] stream   stream to enqueue prefetch operation
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemAdvise(dev_ptr, int count, advice, int device):
    """@brief Advise about the usage of a given memory range to HIP.
    @param [in] dev_ptr  pointer to memory to set the advice for
    @param [in] count    size in bytes of the memory range
    @param [in] advice   advice to be applied for the specified memory range
    @param [in] device   device to apply the advice for
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemRangeGetAttribute(data, int data_size, attribute, dev_ptr, int count):
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

def hipMemRangeGetAttributes(int num_attributes, dev_ptr, int count):
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
    cdef int data_sizes
    cdef chip.hipMemRangeAttribute attributes
    pass

def hipStreamAttachMemAsync(stream, dev_ptr, int length, unsigned int flags):
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

def hipMallocAsync(int size, stream):
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

def hipFreeAsync(dev_ptr, stream):
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

def hipMemPoolTrimTo(mem_pool, int min_bytes_to_hold):
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

def hipMemPoolSetAttribute(mem_pool, attr, value):
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

def hipMemPoolGetAttribute(mem_pool, attr, value):
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

def hipMemPoolSetAccess(mem_pool, desc_list, int count):
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

def hipMemPoolGetAccess(mem_pool, location):
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
    cdef chip.hipMemAccessFlags flags
    pass

def hipMemPoolCreate(pool_props):
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
    mem_pool = ihipMemPoolHandle_t.from_ptr(NULL,owner=True)
    pass

def hipMemPoolDestroy(mem_pool):
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

def hipMallocFromPoolAsync(int size, mem_pool, stream):
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

def hipMemPoolExportToShareableHandle(shared_handle, mem_pool, handle_type, unsigned int flags):
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

def hipMemPoolImportFromShareableHandle(shared_handle, handle_type, unsigned int flags):
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
    mem_pool = ihipMemPoolHandle_t.from_ptr(NULL,owner=True)
    pass

def hipMemPoolExportPointer(export_data, dev_ptr):
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

def hipMemPoolImportPointer(mem_pool, export_data):
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

def hipHostAlloc(int size, unsigned int flags):
    """@brief Allocate device accessible page locked host memory [Deprecated]
    @param[out] ptr Pointer to the allocated host pinned memory
    @param[in]  size Requested memory size
    @param[in]  flags Type of host memory allocation
    If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    @return #hipSuccess, #hipErrorOutOfMemory
    @deprecated use hipHostMalloc() instead
    """
    pass

def hipHostGetDevicePointer(hstPtr, unsigned int flags):
    """@brief Get Device pointer from Host Pointer allocated through hipHostMalloc
    @param[out] dstPtr Device Pointer mapped to passed host pointer
    @param[in]  hstPtr Host Pointer allocated through hipHostMalloc
    @param[in]  flags Flags to be passed for extension
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
    @see hipSetDeviceFlags, hipHostMalloc
    """
    pass

def hipHostGetFlags(hostPtr):
    """@brief Return flags associated with host pointer
    @param[out] flagsPtr Memory location to store flags
    @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
    @return #hipSuccess, #hipErrorInvalidValue
    @see hipHostMalloc
    """
    cdef unsigned int flagsPtr
    pass

def hipHostRegister(hostPtr, int sizeBytes, unsigned int flags):
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

def hipHostUnregister(hostPtr):
    """@brief Un-register host pointer
    @param[in] hostPtr Host pointer previously registered with #hipHostRegister
    @return Error code
    @see hipHostRegister
    """
    pass

def hipMallocPitch(int width, int height):
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
    cdef int pitch
    pass

def hipMemAllocPitch(int widthInBytes, int height, unsigned int elementSizeBytes):
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
    cdef int pitch
    pass

def hipFree(ptr):
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

def hipFreeHost(ptr):
    """@brief Free memory allocated by the hcc hip host memory allocation API.  [Deprecated]
    @param[in] ptr Pointer to memory to be freed
    @return #hipSuccess,
    #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
     hipMalloc)
    @deprecated use hipHostFree() instead
    """
    pass

def hipHostFree(ptr):
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

def hipMemcpy(dst, src, int sizeBytes, kind):
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

def hipMemcpyWithStream(dst, src, int sizeBytes, kind, stream):
    """
    """
    pass

def hipMemcpyHtoD(dst, src, int sizeBytes):
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

def hipMemcpyDtoH(dst, src, int sizeBytes):
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

def hipMemcpyDtoD(dst, src, int sizeBytes):
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

def hipMemcpyHtoDAsync(dst, src, int sizeBytes, stream):
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

def hipMemcpyDtoHAsync(dst, src, int sizeBytes, stream):
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

def hipMemcpyDtoDAsync(dst, src, int sizeBytes, stream):
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

def hipModuleGetGlobal(hmod, const char * name):
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
    cdef int bytes
    pass

def hipGetSymbolAddress(symbol):
    """@brief Gets device pointer associated with symbol on the device.
    @param[out]  devPtr  pointer to the device associated the symbole
    @param[in]   symbol  pointer to the symbole of the device
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipGetSymbolSize(symbol):
    """@brief Gets the size of the given symbol on the device.
    @param[in]   symbol  pointer to the device symbole
    @param[out]  size  pointer to the size
    @return #hipSuccess, #hipErrorInvalidValue
    """
    cdef int size
    pass

def hipMemcpyToSymbol(symbol, src, int sizeBytes, int offset, kind):
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

def hipMemcpyToSymbolAsync(symbol, src, int sizeBytes, int offset, kind, stream):
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

def hipMemcpyFromSymbol(dst, symbol, int sizeBytes, int offset, kind):
    """@brief Copies data from the given symbol on the device.
    @param[out]  dptr  Returns pointer to destinition memory address
    @param[in]   symbol  pointer to the symbole address on the device
    @param[in]   sizeBytes  size in bytes to copy
    @param[in]   offset  offset in bytes from the start of symbole
    @param[in]   kind  type of memory transfer
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMemcpyFromSymbolAsync(dst, symbol, int sizeBytes, int offset, kind, stream):
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

def hipMemcpyAsync(dst, src, int sizeBytes, kind, stream):
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

def hipMemset(dst, int value, int sizeBytes):
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    byte value value.
    @param[out] dst Data being filled
    @param[in]  constant value to be set
    @param[in]  sizeBytes Data size in bytes
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD8(dest, unsigned char value, int count):
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    byte value value.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD8Async(dest, unsigned char value, int count, stream):
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

def hipMemsetD16(dest, unsigned short value, int count):
    """@brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
    short value value.
    @param[out] dst Data ptr to be filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetD16Async(dest, unsigned short value, int count, stream):
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

def hipMemsetD32(dest, int value, int count):
    """@brief Fills the memory area pointed to by dest with the constant integer
    value for specified number of times.
    @param[out] dst Data being filled
    @param[in]  constant value to be set
    @param[in]  number of values to be set
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    """
    pass

def hipMemsetAsync(dst, int value, int sizeBytes, stream):
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

def hipMemsetD32Async(dst, int value, int count, stream):
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

def hipMemset2D(dst, int pitch, int value, int width, int height):
    """@brief Fills the memory area pointed to by dst with the constant value.
    @param[out] dst Pointer to device memory
    @param[in]  pitch - data size in bytes
    @param[in]  value - constant value to be set
    @param[in]  width
    @param[in]  height
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset2DAsync(dst, int pitch, int value, int width, int height, stream):
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

def hipMemset3D(pitchedDevPtr, int value, extent):
    """@brief Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
    @param[in] pitchedDevPtr
    @param[in]  value - constant value to be set
    @param[in]  extent
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemset3DAsync(pitchedDevPtr, int value, extent, stream):
    """@brief Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
    @param[in] pitchedDevPtr
    @param[in]  value - constant value to be set
    @param[in]  extent
    @param[in]  stream
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
    """
    pass

def hipMemGetInfo():
    """@brief Query memory info.
    Return snapshot of free memory, and total allocatable memory on the device.
    Returns in *free a snapshot of the current free memory.
    @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
    @warning On HCC, the free memory only accounts for memory allocated by this process and may be
    optimistic.
    """
    cdef int free
    cdef int total
    hipMemGetInfo_____retval = hipError_t(chip.hipMemGetInfo(&free,&total))    # fully specified
    return (hipMemGetInfo_____retval,free,total)


def hipMemPtrGetInfo(ptr):
    """
    """
    cdef int size
    pass

def hipMallocArray(desc, int width, int height, unsigned int flags):
    """@brief Allocate an array on the device.
    @param[out]  array  Pointer to allocated array in device memory
    @param[in]   desc   Requested channel format
    @param[in]   width  Requested array allocation width
    @param[in]   height Requested array allocation height
    @param[in]   flags  Requested properties of allocated array
    @return      #hipSuccess, #hipErrorOutOfMemory
    @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
    """
    array = hipArray.from_ptr(NULL,owner=True)
    pass

def hipArrayCreate(pAllocateArray):
    """
    """
    pHandle = hipArray.from_ptr(NULL,owner=True)
    pass

def hipArrayDestroy(array):
    """
    """
    pass

def hipArray3DCreate(pAllocateArray):
    """
    """
    array = hipArray.from_ptr(NULL,owner=True)
    pass

def hipMalloc3D(pitchedDevPtr, extent):
    """
    """
    pass

def hipFreeArray(array):
    """@brief Frees an array on the device.
    @param[in]  array  Pointer to array to free
    @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
    @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
    """
    pass

def hipFreeMipmappedArray(mipmappedArray):
    """@brief Frees a mipmapped array on the device
    @param[in] mipmappedArray - Pointer to mipmapped array to free
    @return #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipMalloc3DArray(desc, extent, unsigned int flags):
    """@brief Allocate an array on the device.
    @param[out]  array  Pointer to allocated array in device memory
    @param[in]   desc   Requested channel format
    @param[in]   extent Requested array allocation width, height and depth
    @param[in]   flags  Requested properties of allocated array
    @return      #hipSuccess, #hipErrorOutOfMemory
    @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
    """
    array = hipArray.from_ptr(NULL,owner=True)
    pass

def hipMallocMipmappedArray(desc, extent, unsigned int numLevels, unsigned int flags):
    """@brief Allocate a mipmapped array on the device
    @param[out] mipmappedArray  - Pointer to allocated mipmapped array in device memory
    @param[in]  desc            - Requested channel format
    @param[in]  extent          - Requested allocation size (width field in elements)
    @param[in]  numLevels       - Number of mipmap levels to allocate
    @param[in]  flags           - Flags for extensions
    @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    """
    mipmappedArray = hipMipmappedArray.from_ptr(NULL,owner=True)
    pass

def hipGetMipmappedArrayLevel(mipmappedArray, unsigned int level):
    """@brief Gets a mipmap level of a HIP mipmapped array
    @param[out] levelArray     - Returned mipmap level HIP array
    @param[in]  mipmappedArray - HIP mipmapped array
    @param[in]  level          - Mipmap level
    @return #hipSuccess, #hipErrorInvalidValue
    """
    levelArray = hipArray.from_ptr(NULL,owner=True)
    pass

def hipMemcpy2D(dst, int dpitch, src, int spitch, int width, int height, kind):
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

def hipMemcpyParam2D(pCopy):
    """@brief Copies memory for 2D arrays.
    @param[in]   pCopy Parameters for the memory copy
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    hipMemcpyToSymbol, hipMemcpyAsync
    """
    pass

def hipMemcpyParam2DAsync(pCopy, stream):
    """@brief Copies memory for 2D arrays.
    @param[in]   pCopy Parameters for the memory copy
    @param[in]   stream Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    hipMemcpyToSymbol, hipMemcpyAsync
    """
    pass

def hipMemcpy2DAsync(dst, int dpitch, src, int spitch, int width, int height, kind, stream):
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

def hipMemcpy2DToArray(dst, int wOffset, int hOffset, src, int spitch, int width, int height, kind):
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

def hipMemcpy2DToArrayAsync(dst, int wOffset, int hOffset, src, int spitch, int width, int height, kind, stream):
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

def hipMemcpyToArray(dst, int wOffset, int hOffset, src, int count, kind):
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

def hipMemcpyFromArray(dst, srcArray, int wOffset, int hOffset, int count, kind):
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

def hipMemcpy2DFromArray(dst, int dpitch, src, int wOffset, int hOffset, int width, int height, kind):
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

def hipMemcpy2DFromArrayAsync(dst, int dpitch, src, int wOffset, int hOffset, int width, int height, kind, stream):
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

def hipMemcpyAtoH(dst, srcArray, int srcOffset, int count):
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

def hipMemcpyHtoA(dstArray, int dstOffset, srcHost, int count):
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

def hipMemcpy3D(p):
    """@brief Copies data between host and device.
    @param[in]   p   3D memory copy parameters
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipMemcpy3DAsync(p, stream):
    """@brief Copies data between host and device asynchronously.
    @param[in]   p        3D memory copy parameters
    @param[in]   stream   Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipDrvMemcpy3D(pCopy):
    """@brief Copies data between host and device.
    @param[in]   pCopy   3D memory copy parameters
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipDrvMemcpy3DAsync(pCopy, stream):
    """@brief Copies data between host and device asynchronously.
    @param[in]   pCopy    3D memory copy parameters
    @param[in]   stream   Stream to use
    @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
    #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
    @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
    hipMemcpyAsync
    """
    pass

def hipDeviceCanAccessPeer(int deviceId, int peerDeviceId):
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
    cdef int canAccessPeer
    hipDeviceCanAccessPeer_____retval = hipError_t(chip.hipDeviceCanAccessPeer(&canAccessPeer,deviceId,peerDeviceId))    # fully specified
    return (hipDeviceCanAccessPeer_____retval,canAccessPeer)


def hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags):
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
    hipDeviceEnablePeerAccess_____retval = hipError_t(chip.hipDeviceEnablePeerAccess(peerDeviceId,flags))    # fully specified
    return hipDeviceEnablePeerAccess_____retval


def hipDeviceDisablePeerAccess(int peerDeviceId):
    """@brief Disable direct access from current device's virtual address space to memory allocations
    physically located on a peer device.
    Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
    enabled from the current device.
    @param [in] peerDeviceId
    @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
    """
    hipDeviceDisablePeerAccess_____retval = hipError_t(chip.hipDeviceDisablePeerAccess(peerDeviceId))    # fully specified
    return hipDeviceDisablePeerAccess_____retval


def hipMemGetAddressRange(dptr):
    """@brief Get information on memory allocations.
    @param [out] pbase - BAse pointer address
    @param [out] psize - Size of allocation
    @param [in]  dptr- Device Pointer
    @returns #hipSuccess, #hipErrorInvalidDevicePointer
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    cdef int psize
    pass

def hipMemcpyPeer(dst, int dstDeviceId, src, int srcDeviceId, int sizeBytes):
    """@brief Copies memory from one device to memory on another device.
    @param [out] dst - Destination device pointer.
    @param [in] dstDeviceId - Destination device
    @param [in] src - Source device pointer
    @param [in] srcDeviceId - Source device
    @param [in] sizeBytes - Size of memory copy in bytes
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
    """
    pass

def hipMemcpyPeerAsync(dst, int dstDeviceId, src, int srcDevice, int sizeBytes, stream):
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

def hipCtxCreate(unsigned int flags, hipDevice_t device):
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
    ctx = ihipCtx_t.from_ptr(NULL,owner=True)
    hipCtxCreate_____retval = hipError_t(chip.hipCtxCreate(&ctx._ptr,flags,device))    # fully specified
    return (hipCtxCreate_____retval,ctx)


def hipCtxDestroy(ctx):
    """@brief Destroy a HIP context.
    @param [in] ctx Context to destroy
    @returns #hipSuccess, #hipErrorInvalidValue
    @see hipCtxCreate, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,hipCtxSetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    """
    pass

def hipCtxPopCurrent():
    """@brief Pop the current/default context and return the popped context.
    @param [out] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxSetCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    ctx = ihipCtx_t.from_ptr(NULL,owner=True)
    hipCtxPopCurrent_____retval = hipError_t(chip.hipCtxPopCurrent(&ctx._ptr))    # fully specified
    return (hipCtxPopCurrent_____retval,ctx)


def hipCtxPushCurrent(ctx):
    """@brief Push the context to be set as current/ default context
    @param [in] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    """
    pass

def hipCtxSetCurrent(ctx):
    """@brief Set the passed context as current/default
    @param [in] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
    """
    pass

def hipCtxGetCurrent():
    """@brief Get the handle of the current/ default context
    @param [out] ctx
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    ctx = ihipCtx_t.from_ptr(NULL,owner=True)
    hipCtxGetCurrent_____retval = hipError_t(chip.hipCtxGetCurrent(&ctx._ptr))    # fully specified
    return (hipCtxGetCurrent_____retval,ctx)


def hipCtxGetDevice():
    """@brief Get the handle of the device associated with current/default context
    @param [out] device
    @returns #hipSuccess, #hipErrorInvalidContext
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize
    """
    cdef int device
    hipCtxGetDevice_____retval = hipError_t(chip.hipCtxGetDevice(&device))    # fully specified
    return (hipCtxGetDevice_____retval,device)


def hipCtxGetApiVersion(ctx):
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
    cdef int apiVersion
    pass

def hipCtxGetCacheConfig():
    """@brief Set Cache configuration for a specific function
    @param [out] cacheConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    cdef chip.hipFuncCache_t cacheConfig
    hipCtxGetCacheConfig_____retval = hipError_t(chip.hipCtxGetCacheConfig(&cacheConfig))    # fully specified
    return (hipCtxGetCacheConfig_____retval,hipFuncCache_t(cacheConfig))


def hipCtxSetCacheConfig(cacheConfig):
    """@brief Set L1/Shared cache partition.
    @param [in] cacheConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxSetSharedMemConfig(config):
    """@brief Set Shared memory bank configuration.
    @param [in] sharedMemoryConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pass

def hipCtxGetSharedMemConfig():
    """@brief Get Shared memory bank configuration.
    @param [out] sharedMemoryConfiguration
    @return #hipSuccess
    @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
    ignored on those architectures.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    cdef chip.hipSharedMemConfig pConfig
    hipCtxGetSharedMemConfig_____retval = hipError_t(chip.hipCtxGetSharedMemConfig(&pConfig))    # fully specified
    return (hipCtxGetSharedMemConfig_____retval,hipSharedMemConfig(pConfig))


def hipCtxSynchronize():
    """@brief Blocks until the default context has completed all preceding requested tasks.
    @return #hipSuccess
    @warning This function waits for all streams on the default context to complete execution, and
    then returns.
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxGetDevice
    """
    hipCtxSynchronize_____retval = hipError_t(chip.hipCtxSynchronize())    # fully specified
    return hipCtxSynchronize_____retval


def hipCtxGetFlags():
    """@brief Return flags used for creating default context.
    @param [out] flags
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    cdef unsigned int flags
    hipCtxGetFlags_____retval = hipError_t(chip.hipCtxGetFlags(&flags))    # fully specified
    return (hipCtxGetFlags_____retval,flags)


def hipCtxEnablePeerAccess(peerCtx, unsigned int flags):
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

def hipCtxDisablePeerAccess(peerCtx):
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

def hipDevicePrimaryCtxGetState(hipDevice_t dev):
    """@}
    @brief Get the state of the primary context.
    @param [in] Device to get primary context flags for
    @param [out] Pointer to store flags
    @param [out] Pointer to store context state; 0 = inactive, 1 = active
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    cdef unsigned int flags
    cdef int active
    hipDevicePrimaryCtxGetState_____retval = hipError_t(chip.hipDevicePrimaryCtxGetState(dev,&flags,&active))    # fully specified
    return (hipDevicePrimaryCtxGetState_____retval,flags,active)


def hipDevicePrimaryCtxRelease(hipDevice_t dev):
    """@brief Release the primary context on the GPU.
    @param [in] Device which primary context is released
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    @warning This function return #hipSuccess though doesn't release the primaryCtx by design on
    HIP/HCC path.
    """
    hipDevicePrimaryCtxRelease_____retval = hipError_t(chip.hipDevicePrimaryCtxRelease(dev))    # fully specified
    return hipDevicePrimaryCtxRelease_____retval


def hipDevicePrimaryCtxRetain(hipDevice_t dev):
    """@brief Retain the primary context on the GPU.
    @param [out] Returned context handle of the new context
    @param [in] Device which primary context is released
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    pctx = ihipCtx_t.from_ptr(NULL,owner=True)
    hipDevicePrimaryCtxRetain_____retval = hipError_t(chip.hipDevicePrimaryCtxRetain(&pctx._ptr,dev))    # fully specified
    return (hipDevicePrimaryCtxRetain_____retval,pctx)


def hipDevicePrimaryCtxReset(hipDevice_t dev):
    """@brief Resets the primary context on the GPU.
    @param [in] Device which primary context is reset
    @returns #hipSuccess
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    hipDevicePrimaryCtxReset_____retval = hipError_t(chip.hipDevicePrimaryCtxReset(dev))    # fully specified
    return hipDevicePrimaryCtxReset_____retval


def hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags):
    """@brief Set flags for the primary context.
    @param [in] Device for which the primary context flags are set
    @param [in] New flags for the device
    @returns #hipSuccess, #hipErrorContextAlreadyInUse
    @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
    hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
    """
    hipDevicePrimaryCtxSetFlags_____retval = hipError_t(chip.hipDevicePrimaryCtxSetFlags(dev,flags))    # fully specified
    return hipDevicePrimaryCtxSetFlags_____retval


def hipModuleLoad(const char * fname):
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
    module = ihipModule_t.from_ptr(NULL,owner=True)
    hipModuleLoad_____retval = hipError_t(chip.hipModuleLoad(&module._ptr,fname))    # fully specified
    return (hipModuleLoad_____retval,module)


def hipModuleUnload(module):
    """@brief Frees the module
    @param [in] module
    @returns hipSuccess, hipInvalidValue
    module is freed and the code objects associated with it are destroyed
    """
    pass

def hipModuleGetFunction(module, const char * kname):
    """@brief Function with kname will be extracted if present in module
    @param [in] module
    @param [in] kname
    @param [out] function
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorNotInitialized,
    hipErrorNotFound,
    """
    function = ihipModuleSymbol_t.from_ptr(NULL,owner=True)
    pass

def hipFuncGetAttributes(attr, func):
    """@brief Find out attributes for a given function.
    @param [out] attr
    @param [in] func
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
    """
    pass

def hipFuncGetAttribute(attrib, hfunc):
    """@brief Find out a specific attribute for a given function.
    @param [out] value
    @param [in]  attrib
    @param [in]  hfunc
    @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
    """
    cdef int value
    pass

def hipModuleGetTexRef(hmod, const char * name):
    """@brief returns the handle of the texture reference with the name from the module.
    @param [in] hmod
    @param [in] name
    @param [out] texRef
    @returns hipSuccess, hipErrorNotInitialized, hipErrorNotFound, hipErrorInvalidValue
    """
    texRef = textureReference.from_ptr(NULL,owner=True)
    pass

def hipModuleLoadData(image):
    """@brief builds module from code object which resides in host memory. Image is pointer to that
    location.
    @param [in] image
    @param [out] module
    @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
    """
    module = ihipModule_t.from_ptr(NULL,owner=True)
    pass

def hipModuleLoadDataEx(image, unsigned int numOptions):
    """@brief builds module from code object which resides in host memory. Image is pointer to that
    location. Options are not used. hipModuleLoadData is called.
    @param [in] image
    @param [out] module
    @param [in] number of options
    @param [in] options for JIT
    @param [in] option values for JIT
    @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
    """
    module = ihipModule_t.from_ptr(NULL,owner=True)
    cdef chip.hipJitOption options
    pass

def hipModuleLaunchKernel(f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, stream):
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

def hipLaunchCooperativeKernel(f, gridDim, blockDimX, unsigned int sharedMemBytes, stream):
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

def hipLaunchCooperativeKernelMultiDevice(launchParamsList, int numDevices, unsigned int flags):
    """@brief Launches kernels on multiple devices where thread blocks can cooperate and
    synchronize as they execute.
    @param [in] launchParamsList         List of launch parameters, one per device.
    @param [in] numDevices               Size of the launchParamsList array.
    @param [in] flags                    Flags to control launch behavior.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
    """
    pass

def hipExtLaunchMultiKernelMultiDevice(launchParamsList, int numDevices, unsigned int flags):
    """@brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
    on respective streams before enqueuing any other work on the specified streams from any other threads
    @param [in] hipLaunchParams          List of launch parameters, one per device.
    @param [in] numDevices               Size of the launchParamsList array.
    @param [in] flags                    Flags to control launch behavior.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    """
    pass

def hipModuleOccupancyMaxPotentialBlockSize(f, int dynSharedMemPerBlk, int blockSizeLimit):
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
    cdef int gridSize
    cdef int blockSize
    pass

def hipModuleOccupancyMaxPotentialBlockSizeWithFlags(f, int dynSharedMemPerBlk, int blockSizeLimit, unsigned int flags):
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
    cdef int gridSize
    cdef int blockSize
    pass

def hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(f, int blockSize, int dynSharedMemPerBlk):
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  func             Kernel function (hipFunction) for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    """
    cdef int numBlocks
    pass

def hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(f, int blockSize, int dynSharedMemPerBlk, unsigned int flags):
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  f                Kernel function(hipFunction_t) for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  flags            Extra flags for occupancy calculation (only default supported)
    """
    cdef int numBlocks
    pass

def hipOccupancyMaxActiveBlocksPerMultiprocessor(f, int blockSize, int dynSharedMemPerBlk):
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  func             Kernel function for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    """
    cdef int numBlocks
    pass

def hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(f, int blockSize, int dynSharedMemPerBlk, unsigned int flags):
    """@brief Returns occupancy for a device function.
    @param [out] numBlocks        Returned occupancy
    @param [in]  f                Kernel function for which occupancy is calulated
    @param [in]  blockSize        Block size the kernel is intended to be launched with
    @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
    @param [in]  flags            Extra flags for occupancy calculation (currently ignored)
    """
    cdef int numBlocks
    pass

def hipOccupancyMaxPotentialBlockSize(f, int dynSharedMemPerBlk, int blockSizeLimit):
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
    cdef int gridSize
    cdef int blockSize
    pass

def hipProfilerStart():
    """@brief Start recording of profiling information
    When using this API, start the profiler with profiling disabled.  (--startdisabled)
    @warning : hipProfilerStart API is under development.
    """
    hipProfilerStart_____retval = hipError_t(chip.hipProfilerStart())    # fully specified
    return hipProfilerStart_____retval


def hipProfilerStop():
    """@brief Stop recording of profiling information.
    When using this API, start the profiler with profiling disabled.  (--startdisabled)
    @warning : hipProfilerStop API is under development.
    """
    hipProfilerStop_____retval = hipError_t(chip.hipProfilerStop())    # fully specified
    return hipProfilerStop_____retval


def hipConfigureCall(gridDim, blockDim, int sharedMem, stream):
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

def hipSetupArgument(arg, int size, int offset):
    """@brief Set a kernel argument.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    @param [in] arg    Pointer the argument in host memory.
    @param [in] size   Size of the argument.
    @param [in] offset Offset of the argument on the argument stack.
    """
    pass

def hipLaunchByPtr(func):
    """@brief Launch a kernel.
    @param [in] func Kernel to launch.
    @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
    """
    pass

def hipLaunchKernel(function_address, numBlocks, dimBlocks, int sharedMemBytes, stream):
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

def hipLaunchHostFunc(stream, fn, userData):
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

def hipDrvMemcpy2DUnaligned(pCopy):
    """Copies memory for 2D arrays.
    @param pCopy           - Parameters for the memory copy
    @returns #hipSuccess, #hipErrorInvalidValue
    """
    pass

def hipExtLaunchKernel(function_address, numBlocks, dimBlocks, int sharedMemBytes, stream, startEvent, stopEvent, int flags):
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

def hipBindTextureToMipmappedArray(tex, mipmappedArray, desc):
    """@brief  Binds a mipmapped array to a texture.
    @param [in] tex  pointer to the texture reference to bind
    @param [in] mipmappedArray  memory mipmapped array on the device
    @param [in] desc  opointer to the channel format
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipCreateTextureObject(pResDesc, pTexDesc, pResViewDesc):
    """@brief Creates a texture object.
    @param [out] pTexObject  pointer to the texture object to create
    @param [in] pResDesc  pointer to resource descriptor
    @param [in] pTexDesc  pointer to texture descriptor
    @param [in] pResViewDesc  pointer to resource view descriptor
    @returns hipSuccess, hipErrorInvalidValue, hipErrorNotSupported, hipErrorOutOfMemory
    @note 3D liner filter isn't supported on GFX90A boards, on which the API @p hipCreateTextureObject will
    return hipErrorNotSupported.
    """
    pTexObject = __hip_texture.from_ptr(NULL,owner=True)
    pass

def hipDestroyTextureObject(textureObject):
    """@brief Destroys a texture object.
    @param [in] textureObject  texture object to destroy
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetChannelDesc(desc, array):
    """@brief Gets the channel descriptor in an array.
    @param [in] desc  pointer to channel format descriptor
    @param [out] array  memory array on the device
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetTextureObjectResourceDesc(pResDesc, textureObject):
    """@brief Gets resource descriptor for the texture object.
    @param [out] pResDesc  pointer to resource descriptor
    @param [in] textureObject  texture object
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetTextureObjectResourceViewDesc(pResViewDesc, textureObject):
    """@brief Gets resource view descriptor for the texture object.
    @param [out] pResViewDesc  pointer to resource view descriptor
    @param [in] textureObject  texture object
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipGetTextureObjectTextureDesc(pTexDesc, textureObject):
    """@brief Gets texture descriptor for the texture object.
    @param [out] pTexDesc  pointer to texture descriptor
    @param [in] textureObject  texture object
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexObjectCreate(pResDesc, pTexDesc, pResViewDesc):
    """@brief Creates a texture object.
    @param [out] pTexObject  pointer to texture object to create
    @param [in] pResDesc  pointer to resource descriptor
    @param [in] pTexDesc  pointer to texture descriptor
    @param [in] pResViewDesc  pointer to resource view descriptor
    @returns hipSuccess, hipErrorInvalidValue
    """
    pTexObject = __hip_texture.from_ptr(NULL,owner=True)
    pass

def hipTexObjectDestroy(texObject):
    """@brief Destroys a texture object.
    @param [in] texObject  texture object to destroy
    @returns hipSuccess, hipErrorInvalidValue
    """
    pass

def hipTexObjectGetResourceDesc(pResDesc, texObject):
    """@brief Gets resource descriptor of a texture object.
    @param [out] pResDesc  pointer to resource descriptor
    @param [in] texObject  texture object
    @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    """
    pass

def hipTexObjectGetResourceViewDesc(pResViewDesc, texObject):
    """@brief Gets resource view descriptor of a texture object.
    @param [out] pResViewDesc  pointer to resource view descriptor
    @param [in] texObject  texture object
    @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    """
    pass

def hipTexObjectGetTextureDesc(pTexDesc, texObject):
    """@brief Gets texture descriptor of a texture object.
    @param [out] pTexDesc  pointer to texture descriptor
    @param [in] texObject  texture object
    @returns hipSuccess, hipErrorNotSupported, hipErrorInvalidValue
    """
    pass

def hipGetTextureReference(symbol):
    """@addtogroup TextureD Texture Management [Deprecated]
    @{
    @ingroup Texture
    This section describes the deprecated texture management functions of HIP runtime API.
    @brief Gets the texture reference related with the symbol.
    @param [out] texref  texture reference
    @param [in] symbol  pointer to the symbol related with the texture for the reference
    @returns hipSuccess, hipErrorInvalidValue
    """
    texref = textureReference.from_ptr(NULL,owner=True)
    pass

def hipTexRefSetAddressMode(texRef, int dim, am):
    """
    """
    pass

def hipTexRefSetArray(tex, array, unsigned int flags):
    """
    """
    pass

def hipTexRefSetFilterMode(texRef, fm):
    """
    """
    pass

def hipTexRefSetFlags(texRef, unsigned int Flags):
    """
    """
    pass

def hipTexRefSetFormat(texRef, fmt, int NumPackedComponents):
    """
    """
    pass

def hipBindTexture(tex, devPtr, desc, int size):
    """
    """
    cdef int offset
    pass

def hipBindTexture2D(tex, devPtr, desc, int width, int height, int pitch):
    """
    """
    cdef int offset
    pass

def hipBindTextureToArray(tex, array, desc):
    """
    """
    pass

def hipGetTextureAlignmentOffset(texref):
    """
    """
    cdef int offset
    pass

def hipUnbindTexture(tex):
    """
    """
    pass

def hipTexRefGetAddress(texRef):
    """
    """
    pass

def hipTexRefGetAddressMode(texRef, int dim):
    """
    """
    cdef chip.hipTextureAddressMode pam
    pass

def hipTexRefGetFilterMode(texRef):
    """
    """
    cdef chip.hipTextureFilterMode pfm
    pass

def hipTexRefGetFlags(texRef):
    """
    """
    cdef unsigned int pFlags
    pass

def hipTexRefGetFormat(texRef):
    """
    """
    cdef chip.hipArray_Format pFormat
    cdef int pNumChannels
    pass

def hipTexRefGetMaxAnisotropy(texRef):
    """
    """
    cdef int pmaxAnsio
    pass

def hipTexRefGetMipmapFilterMode(texRef):
    """
    """
    cdef chip.hipTextureFilterMode pfm
    pass

def hipTexRefGetMipmapLevelBias(texRef):
    """
    """
    cdef float pbias
    pass

def hipTexRefGetMipmapLevelClamp(texRef):
    """
    """
    cdef float pminMipmapLevelClamp
    cdef float pmaxMipmapLevelClamp
    pass

def hipTexRefGetMipMappedArray(texRef):
    """
    """
    pArray = hipMipmappedArray.from_ptr(NULL,owner=True)
    pass

def hipTexRefSetAddress(texRef, dptr, int bytes):
    """
    """
    cdef int ByteOffset
    pass

def hipTexRefSetAddress2D(texRef, desc, dptr, int Pitch):
    """
    """
    pass

def hipTexRefSetMaxAnisotropy(texRef, unsigned int maxAniso):
    """
    """
    pass

def hipTexRefSetBorderColor(texRef):
    """
    """
    cdef float pBorderColor
    pass

def hipTexRefSetMipmapFilterMode(texRef, fm):
    """
    """
    pass

def hipTexRefSetMipmapLevelBias(texRef, float bias):
    """
    """
    pass

def hipTexRefSetMipmapLevelClamp(texRef, float minMipMapLevelClamp, float maxMipMapLevelClamp):
    """
    """
    pass

def hipTexRefSetMipmappedArray(texRef, mipmappedArray, unsigned int Flags):
    """
    """
    pass

def hipMipmappedArrayCreate(pMipmappedArrayDesc, unsigned int numMipmapLevels):
    """@addtogroup TextureU Texture Management [Not supported]
    @{
    @ingroup Texture
    This section describes the texture management functions currently unsupported in HIP runtime.
    """
    pHandle = hipMipmappedArray.from_ptr(NULL,owner=True)
    pass

def hipMipmappedArrayDestroy(hMipmappedArray):
    """
    """
    pass

def hipMipmappedArrayGetLevel(hMipMappedArray, unsigned int level):
    """
    """
    pLevelArray = hipArray.from_ptr(NULL,owner=True)
    pass

def hipApiName(uint32_t id):
    """@defgroup Callback Callback Activity APIs
    @{
    This section describes the callback/Activity of HIP runtime API.
    """
    cdef const char * hipApiName_____retval = chip.hipApiName(id)    # fully specified


def hipKernelNameRef(f):
    """
    """
    pass

def hipKernelNameRefByPtr(hostFunction, stream):
    """
    """
    pass

def hipGetStreamDeviceId(stream):
    """
    """
    pass

def hipStreamBeginCapture(stream, mode):
    """@brief Begins graph capture on a stream.
    @param [in] stream - Stream to initiate capture.
    @param [in] mode - Controls the interaction of this capture sequence with other API calls that
    are not safe.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipStreamEndCapture(stream):
    """@brief Ends capture on a stream, returning the captured graph.
    @param [in] stream - Stream to end capture.
    @param [out] pGraph - returns the graph captured.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pGraph = ihipGraph.from_ptr(NULL,owner=True)
    pass

def hipStreamGetCaptureInfo(stream):
    """@brief Get capture status of a stream.
    @param [in] stream - Stream under capture.
    @param [out] pCaptureStatus - returns current status of the capture.
    @param [out] pId - unique ID of the capture.
    @returns #hipSuccess, #hipErrorStreamCaptureImplicit
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    cdef chip.hipStreamCaptureStatus pCaptureStatus
    cdef unsigned long long pId
    pass

def hipStreamGetCaptureInfo_v2(stream):
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
    cdef chip.hipStreamCaptureStatus captureStatus_out
    cdef unsigned long long id_out
    graph_out = ihipGraph.from_ptr(NULL,owner=True)
    cdef int numDependencies_out
    pass

def hipStreamIsCapturing(stream):
    """@brief Get stream's capture state
    @param [in] stream - Stream under capture.
    @param [out] pCaptureStatus - returns current status of the capture.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    cdef chip.hipStreamCaptureStatus pCaptureStatus
    pass

def hipStreamUpdateCaptureDependencies(stream, int numDependencies, unsigned int flags):
    """@brief Update the set of dependencies in a capturing stream
    @param [in] stream - Stream under capture.
    @param [in] dependencies - pointer to an array of nodes to Add/Replace.
    @param [in] numDependencies - size of the array in dependencies.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorIllegalState
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    dependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipThreadExchangeStreamCaptureMode():
    """@brief Swaps the stream capture mode of a thread.
    @param [in] mode - Pointer to mode value to swap with the current mode
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    cdef chip.hipStreamCaptureMode mode
    hipThreadExchangeStreamCaptureMode_____retval = hipError_t(chip.hipThreadExchangeStreamCaptureMode(&mode))    # fully specified
    return (hipThreadExchangeStreamCaptureMode_____retval,hipStreamCaptureMode(mode))


def hipGraphCreate(unsigned int flags):
    """@brief Creates a graph
    @param [out] pGraph - pointer to graph to create.
    @param [in] flags - flags for graph creation, must be 0.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pGraph = ihipGraph.from_ptr(NULL,owner=True)
    hipGraphCreate_____retval = hipError_t(chip.hipGraphCreate(&pGraph._ptr,flags))    # fully specified
    return (hipGraphCreate_____retval,pGraph)


def hipGraphDestroy(graph):
    """@brief Destroys a graph
    @param [in] graph - instance of graph to destroy.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddDependencies(graph, int numDependencies):
    """@brief Adds dependency edges to a graph.
    @param [in] graph - instance of the graph to add dependencies.
    @param [in] from - pointer to the graph nodes with dependenties to add from.
    @param [in] to - pointer to the graph nodes to add dependenties to.
    @param [in] numDependencies - the number of dependencies to add.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    from_ = hipGraphNode.from_ptr(NULL,owner=True)
    to = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphRemoveDependencies(graph, int numDependencies):
    """@brief Removes dependency edges from a graph.
    @param [in] graph - instance of the graph to remove dependencies.
    @param [in] from - Array of nodes that provide the dependencies.
    @param [in] to - Array of dependent nodes.
    @param [in] numDependencies - the number of dependencies to remove.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    from_ = hipGraphNode.from_ptr(NULL,owner=True)
    to = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphGetEdges(graph):
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
    from_ = hipGraphNode.from_ptr(NULL,owner=True)
    to = hipGraphNode.from_ptr(NULL,owner=True)
    cdef int numEdges
    pass

def hipGraphGetNodes(graph):
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
    nodes = hipGraphNode.from_ptr(NULL,owner=True)
    cdef int numNodes
    pass

def hipGraphGetRootNodes(graph):
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
    pRootNodes = hipGraphNode.from_ptr(NULL,owner=True)
    cdef int pNumRootNodes
    pass

def hipGraphNodeGetDependencies(node):
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
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    cdef int pNumDependencies
    pass

def hipGraphNodeGetDependentNodes(node):
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
    pDependentNodes = hipGraphNode.from_ptr(NULL,owner=True)
    cdef int pNumDependentNodes
    pass

def hipGraphNodeGetType(node):
    """@brief Returns a node's type.
    @param [in] node - instance of the graph to add dependencies.
    @param [out] pType - pointer to the return the type
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    cdef chip.hipGraphNodeType pType
    pass

def hipGraphDestroyNode(node):
    """@brief Remove a node from the graph.
    @param [in] node - graph node to remove
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphClone(originalGraph):
    """@brief Clones a graph.
    @param [out] pGraphClone - Returns newly created cloned graph.
    @param [in] originalGraph - original graph to clone from.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pGraphClone = ihipGraph.from_ptr(NULL,owner=True)
    pass

def hipGraphNodeFindInClone(originalNode, clonedGraph):
    """@brief Finds a cloned version of a node.
    @param [out] pNode - Returns the cloned node.
    @param [in] originalNode - original node handle.
    @param [in] clonedGraph - Cloned graph to query.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pNode = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphInstantiate(graph, char * pLogBuffer, int bufferSize):
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
    pGraphExec = hipGraphExec.from_ptr(NULL,owner=True)
    pErrorNode = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphInstantiateWithFlags(graph, unsigned long long flags):
    """@brief Creates an executable graph from a graph.
    @param [out] pGraphExec - pointer to instantiated executable graph that is created.
    @param [in] graph - instance of graph to instantiate.
    @param [in] flags - Flags to control instantiation.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pGraphExec = hipGraphExec.from_ptr(NULL,owner=True)
    pass

def hipGraphLaunch(graphExec, stream):
    """@brief launches an executable graph in a stream
    @param [in] graphExec - instance of executable graph to launch.
    @param [in] stream - instance of stream in which to launch executable graph.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphUpload(graphExec, stream):
    """@brief uploads an executable graph in a stream
    @param [in] graphExec - instance of executable graph to launch.
    @param [in] stream - instance of stream in which to launch executable graph.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecDestroy(graphExec):
    """@brief Destroys an executable graph
    @param [in] pGraphExec - instance of executable graph to destry.
    @returns #hipSuccess.
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecUpdate(hGraphExec, hGraph):
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
    hErrorNode_out = hipGraphNode.from_ptr(NULL,owner=True)
    cdef chip.hipGraphExecUpdateResult updateResult_out
    pass

def hipGraphAddKernelNode(graph, int numDependencies, pNodeParams):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphKernelNodeGetParams(node, pNodeParams):
    """@brief Gets kernel node's parameters.
    @param [in] node - instance of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeSetParams(node, pNodeParams):
    """@brief Sets a kernel node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams):
    """@brief Sets the parameters for a kernel node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the kernel node parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNode(graph, int numDependencies, pCopyParams):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphMemcpyNodeGetParams(node, pNodeParams):
    """@brief Gets a memcpy node's parameters.
    @param [in] node - instance of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemcpyNodeSetParams(node, pNodeParams):
    """@brief Sets a memcpy node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeSetAttribute(hNode, attr, value):
    """@brief Sets a node attribute.
    @param [in] hNode - instance of the node to set parameters to.
    @param [in] attr - the attribute node is set to.
    @param [in] value - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphKernelNodeGetAttribute(hNode, attr, value):
    """@brief Gets a node attribute.
    @param [in] hNode - instance of the node to set parameters to.
    @param [in] attr - the attribute node is set to.
    @param [in] value - const pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams):
    """@brief Sets the parameters for a memcpy node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - const pointer to the kernel node parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddMemcpyNode1D(graph, int numDependencies, dst, src, int count, kind):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphMemcpyNodeSetParams1D(node, dst, src, int count, kind):
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

def hipGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, int count, kind):
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

def hipGraphAddMemcpyNodeFromSymbol(graph, int numDependencies, dst, symbol, int count, int offset, kind):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, int count, int offset, kind):
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

def hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, int count, int offset, kind):
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

def hipGraphAddMemcpyNodeToSymbol(graph, int numDependencies, symbol, src, int count, int offset, kind):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, int count, int offset, kind):
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

def hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, int count, int offset, kind):
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

def hipGraphAddMemsetNode(graph, int numDependencies, pMemsetParams):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphMemsetNodeGetParams(node, pNodeParams):
    """@brief Gets a memset node's parameters.
    @param [in] node - instane of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphMemsetNodeSetParams(node, pNodeParams):
    """@brief Sets a memset node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams):
    """@brief Sets the parameters for a memset node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddHostNode(graph, int numDependencies, pNodeParams):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphHostNodeGetParams(node, pNodeParams):
    """@brief Returns a host node's parameters.
    @param [in] node - instane of the node to get parameters from.
    @param [out] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphHostNodeSetParams(node, pNodeParams):
    """@brief Sets a host node's parameters.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams):
    """@brief Sets the parameters for a host node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - instance of the node to set parameters to.
    @param [in] pNodeParams - pointer to the parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddChildGraphNode(graph, int numDependencies, childGraph):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphChildGraphNodeGetGraph(node):
    """@brief Gets a handle to the embedded graph of a child graph node.
    @param [in] node - instane of the node to get child graph.
    @param [out] pGraph - pointer to get the graph.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pGraph = ihipGraph.from_ptr(NULL,owner=True)
    pass

def hipGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph):
    """@brief Updates node parameters in the child graph node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] node - node from the graph which was used to instantiate graphExec.
    @param [in] childGraph - child graph with updated parameters.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddEmptyNode(graph, int numDependencies):
    """@brief Creates an empty node and adds it to a graph.
    @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
    @param [in] graph - instane of the graph the node is add to.
    @param [in] pDependencies - const pointer to the node dependenties.
    @param [in] numDependencies - the number of dependencies.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphAddEventRecordNode(graph, int numDependencies, event):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphEventRecordNodeGetEvent(node):
    """@brief Returns the event associated with an event record node.
    @param [in] node -  instane of the node to get event from.
    @param [out] event_out - Pointer to return the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    event_out = ihipEvent_t.from_ptr(NULL,owner=True)
    pass

def hipGraphEventRecordNodeSetEvent(node, event):
    """@brief Sets an event record node's event.
    @param [in] node - instane of the node to set event to.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event):
    """@brief Sets the event for an event record node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] hNode - node from the graph which was used to instantiate graphExec.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphAddEventWaitNode(graph, int numDependencies, event):
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
    pGraphNode = hipGraphNode.from_ptr(NULL,owner=True)
    pDependencies = hipGraphNode.from_ptr(NULL,owner=True)
    pass

def hipGraphEventWaitNodeGetEvent(node):
    """@brief Returns the event associated with an event wait node.
    @param [in] node -  instane of the node to get event from.
    @param [out] event_out - Pointer to return the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    event_out = ihipEvent_t.from_ptr(NULL,owner=True)
    pass

def hipGraphEventWaitNodeSetEvent(node, event):
    """@brief Sets an event wait node's event.
    @param [in] node - instane of the node to set event to.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event):
    """@brief Sets the event for an event record node in the given graphExec.
    @param [in] hGraphExec - instance of the executable graph with the node.
    @param [in] hNode - node from the graph which was used to instantiate graphExec.
    @param [in] event - pointer to the event.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceGetGraphMemAttribute(int device, attr, value):
    """@brief Get the mem attribute for graphs.
    @param [in] device - device the attr is get for.
    @param [in] attr - attr to get.
    @param [out] value - value for specific attr.
    @returns #hipSuccess, #hipErrorInvalidDevice
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceSetGraphMemAttribute(int device, attr, value):
    """@brief Set the mem attribute for graphs.
    @param [in] device - device the attr is set for.
    @param [in] attr - attr to set.
    @param [in] value - value for specific attr.
    @returns #hipSuccess, #hipErrorInvalidDevice
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipDeviceGraphMemTrim(int device):
    """@brief Free unused memory on specific device used for graph back to OS.
    @param [in] device - device the memory is used for graphs
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    hipDeviceGraphMemTrim_____retval = hipError_t(chip.hipDeviceGraphMemTrim(device))    # fully specified
    return hipDeviceGraphMemTrim_____retval


def hipUserObjectCreate(ptr, destroy, unsigned int initialRefcount, unsigned int flags):
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
    object_out = hipUserObject.from_ptr(NULL,owner=True)
    pass

def hipUserObjectRelease(object, unsigned int count):
    """@brief Release number of references to resource.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipUserObjectRetain(object, unsigned int count):
    """@brief Retain number of references to resource.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipGraphRetainUserObject(graph, object, unsigned int count, unsigned int flags):
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

def hipGraphReleaseUserObject(graph, object, unsigned int count):
    """@brief Release user object from graphs.
    @param [in] graph - pointer to graph to retain the user object for.
    @param [in] object - pointer to instace of userobj.
    @param [in] count - reference to resource to be retained.
    @returns #hipSuccess, #hipErrorInvalidValue
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemAddressFree(devPtr, int size):
    """@brief Frees an address range reservation made via hipMemAddressReserve
    @param [in] devPtr - starting address of the range.
    @param [in] size - size of the range.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemAddressReserve(int size, int alignment, addr, unsigned long long flags):
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

def hipMemCreate(int size, prop, unsigned long long flags):
    """@brief Creates a memory allocation described by the properties and size
    @param [out] handle - value of the returned handle.
    @param [in] size - size of the allocation.
    @param [in] prop - properties of the allocation.
    @param [in] flags - currently unused, must be zero.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    handle = ihipMemGenericAllocationHandle.from_ptr(NULL,owner=True)
    pass

def hipMemExportToShareableHandle(shareableHandle, handle, handleType, unsigned long long flags):
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

def hipMemGetAccess(location, ptr):
    """@brief Get the access flags set for the given location and ptr.
    @param [out] flags - flags for this location.
    @param [in] location - target location.
    @param [in] ptr - address to check the access flags.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    cdef unsigned long long flags
    pass

def hipMemGetAllocationGranularity(prop, option):
    """@brief Calculates either the minimal or recommended granularity.
    @param [out] granularity - returned granularity.
    @param [in] prop - location properties.
    @param [in] option - determines which granularity to return.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    cdef int granularity
    pass

def hipMemGetAllocationPropertiesFromHandle(prop, handle):
    """@brief Retrieve the property structure of the given handle.
    @param [out] prop - properties of the given handle.
    @param [in] handle - handle to perform the query on.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemImportFromShareableHandle(osHandle, shHandleType):
    """@brief Imports an allocation from a requested shareable handle type.
    @param [out] handle - returned value.
    @param [in] osHandle - shareable handle representing the memory allocation.
    @param [in] shHandleType - handle type.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    handle = ihipMemGenericAllocationHandle.from_ptr(NULL,owner=True)
    pass

def hipMemMap(ptr, int size, int offset, handle, unsigned long long flags):
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

def hipMemMapArrayAsync(mapInfoList, unsigned int count, stream):
    """@brief Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.
    @param [in] mapInfoList - list of hipArrayMapInfo.
    @param [in] count - number of hipArrayMapInfo in mapInfoList.
    @param [in] stream - stream identifier for the stream to use for map or unmap operations.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemRelease(handle):
    """@brief Release a memory handle representing a memory allocation which was previously allocated through hipMemCreate.
    @param [in] handle - handle of the memory allocation.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass

def hipMemRetainAllocationHandle(addr):
    """@brief Returns the allocation handle of the backing memory allocation given the address.
    @param [out] handle - handle representing addr.
    @param [in] addr - address to look up.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    handle = ihipMemGenericAllocationHandle.from_ptr(NULL,owner=True)
    pass

def hipMemSetAccess(ptr, int size, desc, int count):
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

def hipMemUnmap(ptr, int size):
    """@brief Unmap memory allocation of a given address range.
    @param [in] ptr - starting address of the range to unmap.
    @param [in] size - size of the virtual address range.
    @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
    @warning : This API is marked as beta, meaning, while this is feature complete,
    it is still open to changes and may have outstanding issues.
    """
    pass


cdef class GLuint:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef GLuint from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``GLuint`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef GLuint wrapper = GLuint.__new__(GLuint)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class GLenum:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef GLenum from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``GLenum`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef GLenum wrapper = GLenum.__new__(GLenum)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


def hipGLGetDevices(unsigned int hipDeviceCount, deviceList):
    """
    """
    cdef unsigned int pHipDeviceCount
    cdef int pHipDevices
    pass

def hipGraphicsGLRegisterBuffer(GLuint buffer, unsigned int flags):
    """
    """
    resource = _hipGraphicsResource.from_ptr(NULL,owner=True)
    hipGraphicsGLRegisterBuffer_____retval = hipError_t(chip.hipGraphicsGLRegisterBuffer(&resource._ptr,buffer,flags))    # fully specified
    return (hipGraphicsGLRegisterBuffer_____retval,resource)


def hipGraphicsGLRegisterImage(GLuint image, GLenum target, unsigned int flags):
    """
    """
    resource = _hipGraphicsResource.from_ptr(NULL,owner=True)
    hipGraphicsGLRegisterImage_____retval = hipError_t(chip.hipGraphicsGLRegisterImage(&resource._ptr,image,target,flags))    # fully specified
    return (hipGraphicsGLRegisterImage_____retval,resource)


def hipGraphicsMapResources(int count, stream):
    """
    """
    resources = _hipGraphicsResource.from_ptr(NULL,owner=True)
    pass

def hipGraphicsSubResourceGetMappedArray(resource, unsigned int arrayIndex, unsigned int mipLevel):
    """
    """
    array = hipArray.from_ptr(NULL,owner=True)
    pass

def hipGraphicsResourceGetMappedPointer(resource):
    """
    """
    cdef int size
    pass

def hipGraphicsUnmapResources(int count, stream):
    """
    """
    resources = _hipGraphicsResource.from_ptr(NULL,owner=True)
    pass

def hipGraphicsUnregisterResource(resource):
    """
    """
    pass

def hipMemcpy_spt(dst, src, int sizeBytes, kind):
    """
    """
    pass

def hipMemcpyToSymbol_spt(symbol, src, int sizeBytes, int offset, kind):
    """
    """
    pass

def hipMemcpyFromSymbol_spt(dst, symbol, int sizeBytes, int offset, kind):
    """
    """
    pass

def hipMemcpy2D_spt(dst, int dpitch, src, int spitch, int width, int height, kind):
    """
    """
    pass

def hipMemcpy2DFromArray_spt(dst, int dpitch, src, int wOffset, int hOffset, int width, int height, kind):
    """
    """
    pass

def hipMemcpy3D_spt(p):
    """
    """
    pass

def hipMemset_spt(dst, int value, int sizeBytes):
    """
    """
    pass

def hipMemsetAsync_spt(dst, int value, int sizeBytes, stream):
    """
    """
    pass

def hipMemset2D_spt(dst, int pitch, int value, int width, int height):
    """
    """
    pass

def hipMemset2DAsync_spt(dst, int pitch, int value, int width, int height, stream):
    """
    """
    pass

def hipMemset3DAsync_spt(pitchedDevPtr, int value, extent, stream):
    """
    """
    pass

def hipMemset3D_spt(pitchedDevPtr, int value, extent):
    """
    """
    pass

def hipMemcpyAsync_spt(dst, src, int sizeBytes, kind, stream):
    """
    """
    pass

def hipMemcpy3DAsync_spt(p, stream):
    """
    """
    pass

def hipMemcpy2DAsync_spt(dst, int dpitch, src, int spitch, int width, int height, kind, stream):
    """
    """
    pass

def hipMemcpyFromSymbolAsync_spt(dst, symbol, int sizeBytes, int offset, kind, stream):
    """
    """
    pass

def hipMemcpyToSymbolAsync_spt(symbol, src, int sizeBytes, int offset, kind, stream):
    """
    """
    pass

def hipMemcpyFromArray_spt(dst, src, int wOffsetSrc, int hOffset, int count, kind):
    """
    """
    pass

def hipMemcpy2DToArray_spt(dst, int wOffset, int hOffset, src, int spitch, int width, int height, kind):
    """
    """
    pass

def hipMemcpy2DFromArrayAsync_spt(dst, int dpitch, src, int wOffsetSrc, int hOffsetSrc, int width, int height, kind, stream):
    """
    """
    pass

def hipMemcpy2DToArrayAsync_spt(dst, int wOffset, int hOffset, src, int spitch, int width, int height, kind, stream):
    """
    """
    pass

def hipStreamQuery_spt(stream):
    """
    """
    pass

def hipStreamSynchronize_spt(stream):
    """
    """
    pass

def hipStreamGetPriority_spt(stream):
    """
    """
    cdef int priority
    pass

def hipStreamWaitEvent_spt(stream, event, unsigned int flags):
    """
    """
    pass

def hipStreamGetFlags_spt(stream):
    """
    """
    cdef unsigned int flags
    pass

def hipStreamAddCallback_spt(stream, callback, userData, unsigned int flags):
    """
    """
    pass

def hipEventRecord_spt(event, stream):
    """
    """
    pass

def hipLaunchCooperativeKernel_spt(f, gridDim, blockDim, uint32_t sharedMemBytes, hStream):
    """
    """
    pass

def hipLaunchKernel_spt(function_address, numBlocks, dimBlocks, int sharedMemBytes, stream):
    """
    """
    pass

def hipGraphLaunch_spt(graphExec, stream):
    """
    """
    pass

def hipStreamBeginCapture_spt(stream, mode):
    """
    """
    pass

def hipStreamEndCapture_spt(stream):
    """
    """
    pGraph = ihipGraph.from_ptr(NULL,owner=True)
    pass

def hipStreamIsCapturing_spt(stream):
    """
    """
    cdef chip.hipStreamCaptureStatus pCaptureStatus
    pass

def hipStreamGetCaptureInfo_spt(stream):
    """
    """
    cdef chip.hipStreamCaptureStatus pCaptureStatus
    cdef unsigned long long pId
    pass

def hipStreamGetCaptureInfo_v2_spt(stream):
    """
    """
    cdef chip.hipStreamCaptureStatus captureStatus_out
    cdef unsigned long long id_out
    graph_out = ihipGraph.from_ptr(NULL,owner=True)
    cdef int numDependencies_out
    pass

def hipLaunchHostFunc_spt(stream, fn, userData):
    """
    """
    pass