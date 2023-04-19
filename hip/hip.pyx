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
    def get_totalGlobalMem(self, i):
        """Getter for ``totalGlobalMem`` of element ``i``.
        """
        return self._ptr[i].totalGlobalMem
    def set_totalGlobalMem(self, i, int value):
        """Setter for ``totalGlobalMem`` of element ``i``.
        """
        self._ptr[i].totalGlobalMem = value
    @property
    def totalGlobalMem(self):
        """Getter for ``totalGlobalMem``."""
        return self.get_totalGlobalMem(0)
    @totalGlobalMem.setter
    def totalGlobalMem(self, int value):
        """Setter for ``totalGlobalMem``."""
        self.set_totalGlobalMem(0,value)
    def get_sharedMemPerBlock(self, i):
        """Getter for ``sharedMemPerBlock`` of element ``i``.
        """
        return self._ptr[i].sharedMemPerBlock
    def set_sharedMemPerBlock(self, i, int value):
        """Setter for ``sharedMemPerBlock`` of element ``i``.
        """
        self._ptr[i].sharedMemPerBlock = value
    @property
    def sharedMemPerBlock(self):
        """Getter for ``sharedMemPerBlock``."""
        return self.get_sharedMemPerBlock(0)
    @sharedMemPerBlock.setter
    def sharedMemPerBlock(self, int value):
        """Setter for ``sharedMemPerBlock``."""
        self.set_sharedMemPerBlock(0,value)
    def get_regsPerBlock(self, i):
        """Getter for ``regsPerBlock`` of element ``i``.
        """
        return self._ptr[i].regsPerBlock
    def set_regsPerBlock(self, i, int value):
        """Setter for ``regsPerBlock`` of element ``i``.
        """
        self._ptr[i].regsPerBlock = value
    @property
    def regsPerBlock(self):
        """Getter for ``regsPerBlock``."""
        return self.get_regsPerBlock(0)
    @regsPerBlock.setter
    def regsPerBlock(self, int value):
        """Setter for ``regsPerBlock``."""
        self.set_regsPerBlock(0,value)
    def get_warpSize(self, i):
        """Getter for ``warpSize`` of element ``i``.
        """
        return self._ptr[i].warpSize
    def set_warpSize(self, i, int value):
        """Setter for ``warpSize`` of element ``i``.
        """
        self._ptr[i].warpSize = value
    @property
    def warpSize(self):
        """Getter for ``warpSize``."""
        return self.get_warpSize(0)
    @warpSize.setter
    def warpSize(self, int value):
        """Setter for ``warpSize``."""
        self.set_warpSize(0,value)
    def get_maxThreadsPerBlock(self, i):
        """Getter for ``maxThreadsPerBlock`` of element ``i``.
        """
        return self._ptr[i].maxThreadsPerBlock
    def set_maxThreadsPerBlock(self, i, int value):
        """Setter for ``maxThreadsPerBlock`` of element ``i``.
        """
        self._ptr[i].maxThreadsPerBlock = value
    @property
    def maxThreadsPerBlock(self):
        """Getter for ``maxThreadsPerBlock``."""
        return self.get_maxThreadsPerBlock(0)
    @maxThreadsPerBlock.setter
    def maxThreadsPerBlock(self, int value):
        """Setter for ``maxThreadsPerBlock``."""
        self.set_maxThreadsPerBlock(0,value)
    def get_clockRate(self, i):
        """Getter for ``clockRate`` of element ``i``.
        """
        return self._ptr[i].clockRate
    def set_clockRate(self, i, int value):
        """Setter for ``clockRate`` of element ``i``.
        """
        self._ptr[i].clockRate = value
    @property
    def clockRate(self):
        """Getter for ``clockRate``."""
        return self.get_clockRate(0)
    @clockRate.setter
    def clockRate(self, int value):
        """Setter for ``clockRate``."""
        self.set_clockRate(0,value)
    def get_memoryClockRate(self, i):
        """Getter for ``memoryClockRate`` of element ``i``.
        """
        return self._ptr[i].memoryClockRate
    def set_memoryClockRate(self, i, int value):
        """Setter for ``memoryClockRate`` of element ``i``.
        """
        self._ptr[i].memoryClockRate = value
    @property
    def memoryClockRate(self):
        """Getter for ``memoryClockRate``."""
        return self.get_memoryClockRate(0)
    @memoryClockRate.setter
    def memoryClockRate(self, int value):
        """Setter for ``memoryClockRate``."""
        self.set_memoryClockRate(0,value)
    def get_memoryBusWidth(self, i):
        """Getter for ``memoryBusWidth`` of element ``i``.
        """
        return self._ptr[i].memoryBusWidth
    def set_memoryBusWidth(self, i, int value):
        """Setter for ``memoryBusWidth`` of element ``i``.
        """
        self._ptr[i].memoryBusWidth = value
    @property
    def memoryBusWidth(self):
        """Getter for ``memoryBusWidth``."""
        return self.get_memoryBusWidth(0)
    @memoryBusWidth.setter
    def memoryBusWidth(self, int value):
        """Setter for ``memoryBusWidth``."""
        self.set_memoryBusWidth(0,value)
    def get_totalConstMem(self, i):
        """Getter for ``totalConstMem`` of element ``i``.
        """
        return self._ptr[i].totalConstMem
    def set_totalConstMem(self, i, int value):
        """Setter for ``totalConstMem`` of element ``i``.
        """
        self._ptr[i].totalConstMem = value
    @property
    def totalConstMem(self):
        """Getter for ``totalConstMem``."""
        return self.get_totalConstMem(0)
    @totalConstMem.setter
    def totalConstMem(self, int value):
        """Setter for ``totalConstMem``."""
        self.set_totalConstMem(0,value)
    def get_major(self, i):
        """Getter for ``major`` of element ``i``.
        """
        return self._ptr[i].major
    def set_major(self, i, int value):
        """Setter for ``major`` of element ``i``.
        """
        self._ptr[i].major = value
    @property
    def major(self):
        """Getter for ``major``."""
        return self.get_major(0)
    @major.setter
    def major(self, int value):
        """Setter for ``major``."""
        self.set_major(0,value)
    def get_minor(self, i):
        """Getter for ``minor`` of element ``i``.
        """
        return self._ptr[i].minor
    def set_minor(self, i, int value):
        """Setter for ``minor`` of element ``i``.
        """
        self._ptr[i].minor = value
    @property
    def minor(self):
        """Getter for ``minor``."""
        return self.get_minor(0)
    @minor.setter
    def minor(self, int value):
        """Setter for ``minor``."""
        self.set_minor(0,value)
    def get_multiProcessorCount(self, i):
        """Getter for ``multiProcessorCount`` of element ``i``.
        """
        return self._ptr[i].multiProcessorCount
    def set_multiProcessorCount(self, i, int value):
        """Setter for ``multiProcessorCount`` of element ``i``.
        """
        self._ptr[i].multiProcessorCount = value
    @property
    def multiProcessorCount(self):
        """Getter for ``multiProcessorCount``."""
        return self.get_multiProcessorCount(0)
    @multiProcessorCount.setter
    def multiProcessorCount(self, int value):
        """Setter for ``multiProcessorCount``."""
        self.set_multiProcessorCount(0,value)
    def get_l2CacheSize(self, i):
        """Getter for ``l2CacheSize`` of element ``i``.
        """
        return self._ptr[i].l2CacheSize
    def set_l2CacheSize(self, i, int value):
        """Setter for ``l2CacheSize`` of element ``i``.
        """
        self._ptr[i].l2CacheSize = value
    @property
    def l2CacheSize(self):
        """Getter for ``l2CacheSize``."""
        return self.get_l2CacheSize(0)
    @l2CacheSize.setter
    def l2CacheSize(self, int value):
        """Setter for ``l2CacheSize``."""
        self.set_l2CacheSize(0,value)
    def get_maxThreadsPerMultiProcessor(self, i):
        """Getter for ``maxThreadsPerMultiProcessor`` of element ``i``.
        """
        return self._ptr[i].maxThreadsPerMultiProcessor
    def set_maxThreadsPerMultiProcessor(self, i, int value):
        """Setter for ``maxThreadsPerMultiProcessor`` of element ``i``.
        """
        self._ptr[i].maxThreadsPerMultiProcessor = value
    @property
    def maxThreadsPerMultiProcessor(self):
        """Getter for ``maxThreadsPerMultiProcessor``."""
        return self.get_maxThreadsPerMultiProcessor(0)
    @maxThreadsPerMultiProcessor.setter
    def maxThreadsPerMultiProcessor(self, int value):
        """Setter for ``maxThreadsPerMultiProcessor``."""
        self.set_maxThreadsPerMultiProcessor(0,value)
    def get_computeMode(self, i):
        """Getter for ``computeMode`` of element ``i``.
        """
        return self._ptr[i].computeMode
    def set_computeMode(self, i, int value):
        """Setter for ``computeMode`` of element ``i``.
        """
        self._ptr[i].computeMode = value
    @property
    def computeMode(self):
        """Getter for ``computeMode``."""
        return self.get_computeMode(0)
    @computeMode.setter
    def computeMode(self, int value):
        """Setter for ``computeMode``."""
        self.set_computeMode(0,value)
    def get_clockInstructionRate(self, i):
        """Getter for ``clockInstructionRate`` of element ``i``.
        """
        return self._ptr[i].clockInstructionRate
    def set_clockInstructionRate(self, i, int value):
        """Setter for ``clockInstructionRate`` of element ``i``.
        """
        self._ptr[i].clockInstructionRate = value
    @property
    def clockInstructionRate(self):
        """Getter for ``clockInstructionRate``."""
        return self.get_clockInstructionRate(0)
    @clockInstructionRate.setter
    def clockInstructionRate(self, int value):
        """Setter for ``clockInstructionRate``."""
        self.set_clockInstructionRate(0,value)
    def get_arch(self, i):
        """Getter for ``arch`` of element ``i``.
        """
        return hipDeviceArch_t.from_ptr(&self._ptr[i].arch)
    @property
    def arch(self):
        """Getter for ``arch``."""
        return self.get_arch(0)
    def get_concurrentKernels(self, i):
        """Getter for ``concurrentKernels`` of element ``i``.
        """
        return self._ptr[i].concurrentKernels
    def set_concurrentKernels(self, i, int value):
        """Setter for ``concurrentKernels`` of element ``i``.
        """
        self._ptr[i].concurrentKernels = value
    @property
    def concurrentKernels(self):
        """Getter for ``concurrentKernels``."""
        return self.get_concurrentKernels(0)
    @concurrentKernels.setter
    def concurrentKernels(self, int value):
        """Setter for ``concurrentKernels``."""
        self.set_concurrentKernels(0,value)
    def get_pciDomainID(self, i):
        """Getter for ``pciDomainID`` of element ``i``.
        """
        return self._ptr[i].pciDomainID
    def set_pciDomainID(self, i, int value):
        """Setter for ``pciDomainID`` of element ``i``.
        """
        self._ptr[i].pciDomainID = value
    @property
    def pciDomainID(self):
        """Getter for ``pciDomainID``."""
        return self.get_pciDomainID(0)
    @pciDomainID.setter
    def pciDomainID(self, int value):
        """Setter for ``pciDomainID``."""
        self.set_pciDomainID(0,value)
    def get_pciBusID(self, i):
        """Getter for ``pciBusID`` of element ``i``.
        """
        return self._ptr[i].pciBusID
    def set_pciBusID(self, i, int value):
        """Setter for ``pciBusID`` of element ``i``.
        """
        self._ptr[i].pciBusID = value
    @property
    def pciBusID(self):
        """Getter for ``pciBusID``."""
        return self.get_pciBusID(0)
    @pciBusID.setter
    def pciBusID(self, int value):
        """Setter for ``pciBusID``."""
        self.set_pciBusID(0,value)
    def get_pciDeviceID(self, i):
        """Getter for ``pciDeviceID`` of element ``i``.
        """
        return self._ptr[i].pciDeviceID
    def set_pciDeviceID(self, i, int value):
        """Setter for ``pciDeviceID`` of element ``i``.
        """
        self._ptr[i].pciDeviceID = value
    @property
    def pciDeviceID(self):
        """Getter for ``pciDeviceID``."""
        return self.get_pciDeviceID(0)
    @pciDeviceID.setter
    def pciDeviceID(self, int value):
        """Setter for ``pciDeviceID``."""
        self.set_pciDeviceID(0,value)
    def get_maxSharedMemoryPerMultiProcessor(self, i):
        """Getter for ``maxSharedMemoryPerMultiProcessor`` of element ``i``.
        """
        return self._ptr[i].maxSharedMemoryPerMultiProcessor
    def set_maxSharedMemoryPerMultiProcessor(self, i, int value):
        """Setter for ``maxSharedMemoryPerMultiProcessor`` of element ``i``.
        """
        self._ptr[i].maxSharedMemoryPerMultiProcessor = value
    @property
    def maxSharedMemoryPerMultiProcessor(self):
        """Getter for ``maxSharedMemoryPerMultiProcessor``."""
        return self.get_maxSharedMemoryPerMultiProcessor(0)
    @maxSharedMemoryPerMultiProcessor.setter
    def maxSharedMemoryPerMultiProcessor(self, int value):
        """Setter for ``maxSharedMemoryPerMultiProcessor``."""
        self.set_maxSharedMemoryPerMultiProcessor(0,value)
    def get_isMultiGpuBoard(self, i):
        """Getter for ``isMultiGpuBoard`` of element ``i``.
        """
        return self._ptr[i].isMultiGpuBoard
    def set_isMultiGpuBoard(self, i, int value):
        """Setter for ``isMultiGpuBoard`` of element ``i``.
        """
        self._ptr[i].isMultiGpuBoard = value
    @property
    def isMultiGpuBoard(self):
        """Getter for ``isMultiGpuBoard``."""
        return self.get_isMultiGpuBoard(0)
    @isMultiGpuBoard.setter
    def isMultiGpuBoard(self, int value):
        """Setter for ``isMultiGpuBoard``."""
        self.set_isMultiGpuBoard(0,value)
    def get_canMapHostMemory(self, i):
        """Getter for ``canMapHostMemory`` of element ``i``.
        """
        return self._ptr[i].canMapHostMemory
    def set_canMapHostMemory(self, i, int value):
        """Setter for ``canMapHostMemory`` of element ``i``.
        """
        self._ptr[i].canMapHostMemory = value
    @property
    def canMapHostMemory(self):
        """Getter for ``canMapHostMemory``."""
        return self.get_canMapHostMemory(0)
    @canMapHostMemory.setter
    def canMapHostMemory(self, int value):
        """Setter for ``canMapHostMemory``."""
        self.set_canMapHostMemory(0,value)
    def get_gcnArch(self, i):
        """Getter for ``gcnArch`` of element ``i``.
        """
        return self._ptr[i].gcnArch
    def set_gcnArch(self, i, int value):
        """Setter for ``gcnArch`` of element ``i``.
        """
        self._ptr[i].gcnArch = value
    @property
    def gcnArch(self):
        """Getter for ``gcnArch``."""
        return self.get_gcnArch(0)
    @gcnArch.setter
    def gcnArch(self, int value):
        """Setter for ``gcnArch``."""
        self.set_gcnArch(0,value)
    def get_integrated(self, i):
        """Getter for ``integrated`` of element ``i``.
        """
        return self._ptr[i].integrated
    def set_integrated(self, i, int value):
        """Setter for ``integrated`` of element ``i``.
        """
        self._ptr[i].integrated = value
    @property
    def integrated(self):
        """Getter for ``integrated``."""
        return self.get_integrated(0)
    @integrated.setter
    def integrated(self, int value):
        """Setter for ``integrated``."""
        self.set_integrated(0,value)
    def get_cooperativeLaunch(self, i):
        """Getter for ``cooperativeLaunch`` of element ``i``.
        """
        return self._ptr[i].cooperativeLaunch
    def set_cooperativeLaunch(self, i, int value):
        """Setter for ``cooperativeLaunch`` of element ``i``.
        """
        self._ptr[i].cooperativeLaunch = value
    @property
    def cooperativeLaunch(self):
        """Getter for ``cooperativeLaunch``."""
        return self.get_cooperativeLaunch(0)
    @cooperativeLaunch.setter
    def cooperativeLaunch(self, int value):
        """Setter for ``cooperativeLaunch``."""
        self.set_cooperativeLaunch(0,value)
    def get_cooperativeMultiDeviceLaunch(self, i):
        """Getter for ``cooperativeMultiDeviceLaunch`` of element ``i``.
        """
        return self._ptr[i].cooperativeMultiDeviceLaunch
    def set_cooperativeMultiDeviceLaunch(self, i, int value):
        """Setter for ``cooperativeMultiDeviceLaunch`` of element ``i``.
        """
        self._ptr[i].cooperativeMultiDeviceLaunch = value
    @property
    def cooperativeMultiDeviceLaunch(self):
        """Getter for ``cooperativeMultiDeviceLaunch``."""
        return self.get_cooperativeMultiDeviceLaunch(0)
    @cooperativeMultiDeviceLaunch.setter
    def cooperativeMultiDeviceLaunch(self, int value):
        """Setter for ``cooperativeMultiDeviceLaunch``."""
        self.set_cooperativeMultiDeviceLaunch(0,value)
    def get_maxTexture1DLinear(self, i):
        """Getter for ``maxTexture1DLinear`` of element ``i``.
        """
        return self._ptr[i].maxTexture1DLinear
    def set_maxTexture1DLinear(self, i, int value):
        """Setter for ``maxTexture1DLinear`` of element ``i``.
        """
        self._ptr[i].maxTexture1DLinear = value
    @property
    def maxTexture1DLinear(self):
        """Getter for ``maxTexture1DLinear``."""
        return self.get_maxTexture1DLinear(0)
    @maxTexture1DLinear.setter
    def maxTexture1DLinear(self, int value):
        """Setter for ``maxTexture1DLinear``."""
        self.set_maxTexture1DLinear(0,value)
    def get_maxTexture1D(self, i):
        """Getter for ``maxTexture1D`` of element ``i``.
        """
        return self._ptr[i].maxTexture1D
    def set_maxTexture1D(self, i, int value):
        """Setter for ``maxTexture1D`` of element ``i``.
        """
        self._ptr[i].maxTexture1D = value
    @property
    def maxTexture1D(self):
        """Getter for ``maxTexture1D``."""
        return self.get_maxTexture1D(0)
    @maxTexture1D.setter
    def maxTexture1D(self, int value):
        """Setter for ``maxTexture1D``."""
        self.set_maxTexture1D(0,value)
    def get_memPitch(self, i):
        """Getter for ``memPitch`` of element ``i``.
        """
        return self._ptr[i].memPitch
    def set_memPitch(self, i, int value):
        """Setter for ``memPitch`` of element ``i``.
        """
        self._ptr[i].memPitch = value
    @property
    def memPitch(self):
        """Getter for ``memPitch``."""
        return self.get_memPitch(0)
    @memPitch.setter
    def memPitch(self, int value):
        """Setter for ``memPitch``."""
        self.set_memPitch(0,value)
    def get_textureAlignment(self, i):
        """Getter for ``textureAlignment`` of element ``i``.
        """
        return self._ptr[i].textureAlignment
    def set_textureAlignment(self, i, int value):
        """Setter for ``textureAlignment`` of element ``i``.
        """
        self._ptr[i].textureAlignment = value
    @property
    def textureAlignment(self):
        """Getter for ``textureAlignment``."""
        return self.get_textureAlignment(0)
    @textureAlignment.setter
    def textureAlignment(self, int value):
        """Setter for ``textureAlignment``."""
        self.set_textureAlignment(0,value)
    def get_texturePitchAlignment(self, i):
        """Getter for ``texturePitchAlignment`` of element ``i``.
        """
        return self._ptr[i].texturePitchAlignment
    def set_texturePitchAlignment(self, i, int value):
        """Setter for ``texturePitchAlignment`` of element ``i``.
        """
        self._ptr[i].texturePitchAlignment = value
    @property
    def texturePitchAlignment(self):
        """Getter for ``texturePitchAlignment``."""
        return self.get_texturePitchAlignment(0)
    @texturePitchAlignment.setter
    def texturePitchAlignment(self, int value):
        """Setter for ``texturePitchAlignment``."""
        self.set_texturePitchAlignment(0,value)
    def get_kernelExecTimeoutEnabled(self, i):
        """Getter for ``kernelExecTimeoutEnabled`` of element ``i``.
        """
        return self._ptr[i].kernelExecTimeoutEnabled
    def set_kernelExecTimeoutEnabled(self, i, int value):
        """Setter for ``kernelExecTimeoutEnabled`` of element ``i``.
        """
        self._ptr[i].kernelExecTimeoutEnabled = value
    @property
    def kernelExecTimeoutEnabled(self):
        """Getter for ``kernelExecTimeoutEnabled``."""
        return self.get_kernelExecTimeoutEnabled(0)
    @kernelExecTimeoutEnabled.setter
    def kernelExecTimeoutEnabled(self, int value):
        """Setter for ``kernelExecTimeoutEnabled``."""
        self.set_kernelExecTimeoutEnabled(0,value)
    def get_ECCEnabled(self, i):
        """Getter for ``ECCEnabled`` of element ``i``.
        """
        return self._ptr[i].ECCEnabled
    def set_ECCEnabled(self, i, int value):
        """Setter for ``ECCEnabled`` of element ``i``.
        """
        self._ptr[i].ECCEnabled = value
    @property
    def ECCEnabled(self):
        """Getter for ``ECCEnabled``."""
        return self.get_ECCEnabled(0)
    @ECCEnabled.setter
    def ECCEnabled(self, int value):
        """Setter for ``ECCEnabled``."""
        self.set_ECCEnabled(0,value)
    def get_tccDriver(self, i):
        """Getter for ``tccDriver`` of element ``i``.
        """
        return self._ptr[i].tccDriver
    def set_tccDriver(self, i, int value):
        """Setter for ``tccDriver`` of element ``i``.
        """
        self._ptr[i].tccDriver = value
    @property
    def tccDriver(self):
        """Getter for ``tccDriver``."""
        return self.get_tccDriver(0)
    @tccDriver.setter
    def tccDriver(self, int value):
        """Setter for ``tccDriver``."""
        self.set_tccDriver(0,value)
    def get_cooperativeMultiDeviceUnmatchedFunc(self, i):
        """Getter for ``cooperativeMultiDeviceUnmatchedFunc`` of element ``i``.
        """
        return self._ptr[i].cooperativeMultiDeviceUnmatchedFunc
    def set_cooperativeMultiDeviceUnmatchedFunc(self, i, int value):
        """Setter for ``cooperativeMultiDeviceUnmatchedFunc`` of element ``i``.
        """
        self._ptr[i].cooperativeMultiDeviceUnmatchedFunc = value
    @property
    def cooperativeMultiDeviceUnmatchedFunc(self):
        """Getter for ``cooperativeMultiDeviceUnmatchedFunc``."""
        return self.get_cooperativeMultiDeviceUnmatchedFunc(0)
    @cooperativeMultiDeviceUnmatchedFunc.setter
    def cooperativeMultiDeviceUnmatchedFunc(self, int value):
        """Setter for ``cooperativeMultiDeviceUnmatchedFunc``."""
        self.set_cooperativeMultiDeviceUnmatchedFunc(0,value)
    def get_cooperativeMultiDeviceUnmatchedGridDim(self, i):
        """Getter for ``cooperativeMultiDeviceUnmatchedGridDim`` of element ``i``.
        """
        return self._ptr[i].cooperativeMultiDeviceUnmatchedGridDim
    def set_cooperativeMultiDeviceUnmatchedGridDim(self, i, int value):
        """Setter for ``cooperativeMultiDeviceUnmatchedGridDim`` of element ``i``.
        """
        self._ptr[i].cooperativeMultiDeviceUnmatchedGridDim = value
    @property
    def cooperativeMultiDeviceUnmatchedGridDim(self):
        """Getter for ``cooperativeMultiDeviceUnmatchedGridDim``."""
        return self.get_cooperativeMultiDeviceUnmatchedGridDim(0)
    @cooperativeMultiDeviceUnmatchedGridDim.setter
    def cooperativeMultiDeviceUnmatchedGridDim(self, int value):
        """Setter for ``cooperativeMultiDeviceUnmatchedGridDim``."""
        self.set_cooperativeMultiDeviceUnmatchedGridDim(0,value)
    def get_cooperativeMultiDeviceUnmatchedBlockDim(self, i):
        """Getter for ``cooperativeMultiDeviceUnmatchedBlockDim`` of element ``i``.
        """
        return self._ptr[i].cooperativeMultiDeviceUnmatchedBlockDim
    def set_cooperativeMultiDeviceUnmatchedBlockDim(self, i, int value):
        """Setter for ``cooperativeMultiDeviceUnmatchedBlockDim`` of element ``i``.
        """
        self._ptr[i].cooperativeMultiDeviceUnmatchedBlockDim = value
    @property
    def cooperativeMultiDeviceUnmatchedBlockDim(self):
        """Getter for ``cooperativeMultiDeviceUnmatchedBlockDim``."""
        return self.get_cooperativeMultiDeviceUnmatchedBlockDim(0)
    @cooperativeMultiDeviceUnmatchedBlockDim.setter
    def cooperativeMultiDeviceUnmatchedBlockDim(self, int value):
        """Setter for ``cooperativeMultiDeviceUnmatchedBlockDim``."""
        self.set_cooperativeMultiDeviceUnmatchedBlockDim(0,value)
    def get_cooperativeMultiDeviceUnmatchedSharedMem(self, i):
        """Getter for ``cooperativeMultiDeviceUnmatchedSharedMem`` of element ``i``.
        """
        return self._ptr[i].cooperativeMultiDeviceUnmatchedSharedMem
    def set_cooperativeMultiDeviceUnmatchedSharedMem(self, i, int value):
        """Setter for ``cooperativeMultiDeviceUnmatchedSharedMem`` of element ``i``.
        """
        self._ptr[i].cooperativeMultiDeviceUnmatchedSharedMem = value
    @property
    def cooperativeMultiDeviceUnmatchedSharedMem(self):
        """Getter for ``cooperativeMultiDeviceUnmatchedSharedMem``."""
        return self.get_cooperativeMultiDeviceUnmatchedSharedMem(0)
    @cooperativeMultiDeviceUnmatchedSharedMem.setter
    def cooperativeMultiDeviceUnmatchedSharedMem(self, int value):
        """Setter for ``cooperativeMultiDeviceUnmatchedSharedMem``."""
        self.set_cooperativeMultiDeviceUnmatchedSharedMem(0,value)
    def get_isLargeBar(self, i):
        """Getter for ``isLargeBar`` of element ``i``.
        """
        return self._ptr[i].isLargeBar
    def set_isLargeBar(self, i, int value):
        """Setter for ``isLargeBar`` of element ``i``.
        """
        self._ptr[i].isLargeBar = value
    @property
    def isLargeBar(self):
        """Getter for ``isLargeBar``."""
        return self.get_isLargeBar(0)
    @isLargeBar.setter
    def isLargeBar(self, int value):
        """Setter for ``isLargeBar``."""
        self.set_isLargeBar(0,value)
    def get_asicRevision(self, i):
        """Getter for ``asicRevision`` of element ``i``.
        """
        return self._ptr[i].asicRevision
    def set_asicRevision(self, i, int value):
        """Setter for ``asicRevision`` of element ``i``.
        """
        self._ptr[i].asicRevision = value
    @property
    def asicRevision(self):
        """Getter for ``asicRevision``."""
        return self.get_asicRevision(0)
    @asicRevision.setter
    def asicRevision(self, int value):
        """Setter for ``asicRevision``."""
        self.set_asicRevision(0,value)
    def get_managedMemory(self, i):
        """Getter for ``managedMemory`` of element ``i``.
        """
        return self._ptr[i].managedMemory
    def set_managedMemory(self, i, int value):
        """Setter for ``managedMemory`` of element ``i``.
        """
        self._ptr[i].managedMemory = value
    @property
    def managedMemory(self):
        """Getter for ``managedMemory``."""
        return self.get_managedMemory(0)
    @managedMemory.setter
    def managedMemory(self, int value):
        """Setter for ``managedMemory``."""
        self.set_managedMemory(0,value)
    def get_directManagedMemAccessFromHost(self, i):
        """Getter for ``directManagedMemAccessFromHost`` of element ``i``.
        """
        return self._ptr[i].directManagedMemAccessFromHost
    def set_directManagedMemAccessFromHost(self, i, int value):
        """Setter for ``directManagedMemAccessFromHost`` of element ``i``.
        """
        self._ptr[i].directManagedMemAccessFromHost = value
    @property
    def directManagedMemAccessFromHost(self):
        """Getter for ``directManagedMemAccessFromHost``."""
        return self.get_directManagedMemAccessFromHost(0)
    @directManagedMemAccessFromHost.setter
    def directManagedMemAccessFromHost(self, int value):
        """Setter for ``directManagedMemAccessFromHost``."""
        self.set_directManagedMemAccessFromHost(0,value)
    def get_concurrentManagedAccess(self, i):
        """Getter for ``concurrentManagedAccess`` of element ``i``.
        """
        return self._ptr[i].concurrentManagedAccess
    def set_concurrentManagedAccess(self, i, int value):
        """Setter for ``concurrentManagedAccess`` of element ``i``.
        """
        self._ptr[i].concurrentManagedAccess = value
    @property
    def concurrentManagedAccess(self):
        """Getter for ``concurrentManagedAccess``."""
        return self.get_concurrentManagedAccess(0)
    @concurrentManagedAccess.setter
    def concurrentManagedAccess(self, int value):
        """Setter for ``concurrentManagedAccess``."""
        self.set_concurrentManagedAccess(0,value)
    def get_pageableMemoryAccess(self, i):
        """Getter for ``pageableMemoryAccess`` of element ``i``.
        """
        return self._ptr[i].pageableMemoryAccess
    def set_pageableMemoryAccess(self, i, int value):
        """Setter for ``pageableMemoryAccess`` of element ``i``.
        """
        self._ptr[i].pageableMemoryAccess = value
    @property
    def pageableMemoryAccess(self):
        """Getter for ``pageableMemoryAccess``."""
        return self.get_pageableMemoryAccess(0)
    @pageableMemoryAccess.setter
    def pageableMemoryAccess(self, int value):
        """Setter for ``pageableMemoryAccess``."""
        self.set_pageableMemoryAccess(0,value)
    def get_pageableMemoryAccessUsesHostPageTables(self, i):
        """Getter for ``pageableMemoryAccessUsesHostPageTables`` of element ``i``.
        """
        return self._ptr[i].pageableMemoryAccessUsesHostPageTables
    def set_pageableMemoryAccessUsesHostPageTables(self, i, int value):
        """Setter for ``pageableMemoryAccessUsesHostPageTables`` of element ``i``.
        """
        self._ptr[i].pageableMemoryAccessUsesHostPageTables = value
    @property
    def pageableMemoryAccessUsesHostPageTables(self):
        """Getter for ``pageableMemoryAccessUsesHostPageTables``."""
        return self.get_pageableMemoryAccessUsesHostPageTables(0)
    @pageableMemoryAccessUsesHostPageTables.setter
    def pageableMemoryAccessUsesHostPageTables(self, int value):
        """Setter for ``pageableMemoryAccessUsesHostPageTables``."""
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
        """Getter for ``memoryType`` of element ``i``.
        """
        return hipMemoryType(self._ptr[i].memoryType)
    def set_memoryType(self, i, value):
        """Setter for ``memoryType`` of element ``i``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].memoryType = value.value
    @property
    def memoryType(self):
        """Getter for ``memoryType``."""
        return self.get_memoryType(0)
    @memoryType.setter
    def memoryType(self, value):
        """Setter for ``memoryType``."""
        self.set_memoryType(0,value)
    def get_device(self, i):
        """Getter for ``device`` of element ``i``.
        """
        return self._ptr[i].device
    def set_device(self, i, int value):
        """Setter for ``device`` of element ``i``.
        """
        self._ptr[i].device = value
    @property
    def device(self):
        """Getter for ``device``."""
        return self.get_device(0)
    @device.setter
    def device(self, int value):
        """Setter for ``device``."""
        self.set_device(0,value)
    def get_isManaged(self, i):
        """Getter for ``isManaged`` of element ``i``.
        """
        return self._ptr[i].isManaged
    def set_isManaged(self, i, int value):
        """Setter for ``isManaged`` of element ``i``.
        """
        self._ptr[i].isManaged = value
    @property
    def isManaged(self):
        """Getter for ``isManaged``."""
        return self.get_isManaged(0)
    @isManaged.setter
    def isManaged(self, int value):
        """Setter for ``isManaged``."""
        self.set_isManaged(0,value)
    def get_allocationFlags(self, i):
        """Getter for ``allocationFlags`` of element ``i``.
        """
        return self._ptr[i].allocationFlags
    def set_allocationFlags(self, i, unsigned int value):
        """Setter for ``allocationFlags`` of element ``i``.
        """
        self._ptr[i].allocationFlags = value
    @property
    def allocationFlags(self):
        """Getter for ``allocationFlags``."""
        return self.get_allocationFlags(0)
    @allocationFlags.setter
    def allocationFlags(self, unsigned int value):
        """Setter for ``allocationFlags``."""
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
        """Getter for ``x`` of element ``i``.
        """
        return self._ptr[i].x
    def set_x(self, i, int value):
        """Setter for ``x`` of element ``i``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        """Getter for ``x``."""
        return self.get_x(0)
    @x.setter
    def x(self, int value):
        """Setter for ``x``."""
        self.set_x(0,value)
    def get_y(self, i):
        """Getter for ``y`` of element ``i``.
        """
        return self._ptr[i].y
    def set_y(self, i, int value):
        """Setter for ``y`` of element ``i``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        """Getter for ``y``."""
        return self.get_y(0)
    @y.setter
    def y(self, int value):
        """Setter for ``y``."""
        self.set_y(0,value)
    def get_z(self, i):
        """Getter for ``z`` of element ``i``.
        """
        return self._ptr[i].z
    def set_z(self, i, int value):
        """Setter for ``z`` of element ``i``.
        """
        self._ptr[i].z = value
    @property
    def z(self):
        """Getter for ``z``."""
        return self.get_z(0)
    @z.setter
    def z(self, int value):
        """Setter for ``z``."""
        self.set_z(0,value)
    def get_w(self, i):
        """Getter for ``w`` of element ``i``.
        """
        return self._ptr[i].w
    def set_w(self, i, int value):
        """Setter for ``w`` of element ``i``.
        """
        self._ptr[i].w = value
    @property
    def w(self):
        """Getter for ``w``."""
        return self.get_w(0)
    @w.setter
    def w(self, int value):
        """Setter for ``w``."""
        self.set_w(0,value)
    def get_f(self, i):
        """Getter for ``f`` of element ``i``.
        """
        return hipChannelFormatKind(self._ptr[i].f)
    def set_f(self, i, value):
        """Setter for ``f`` of element ``i``.
        """
        if not isinstance(value, hipChannelFormatKind):
            raise TypeError("'value' must be of type 'hipChannelFormatKind'")
        self._ptr[i].f = value.value
    @property
    def f(self):
        """Getter for ``f``."""
        return self.get_f(0)
    @f.setter
    def f(self, value):
        """Setter for ``f``."""
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
        """Getter for ``Width`` of element ``i``.
        """
        return self._ptr[i].Width
    def set_Width(self, i, int value):
        """Setter for ``Width`` of element ``i``.
        """
        self._ptr[i].Width = value
    @property
    def Width(self):
        """Getter for ``Width``."""
        return self.get_Width(0)
    @Width.setter
    def Width(self, int value):
        """Setter for ``Width``."""
        self.set_Width(0,value)
    def get_Height(self, i):
        """Getter for ``Height`` of element ``i``.
        """
        return self._ptr[i].Height
    def set_Height(self, i, int value):
        """Setter for ``Height`` of element ``i``.
        """
        self._ptr[i].Height = value
    @property
    def Height(self):
        """Getter for ``Height``."""
        return self.get_Height(0)
    @Height.setter
    def Height(self, int value):
        """Setter for ``Height``."""
        self.set_Height(0,value)
    def get_Format(self, i):
        """Getter for ``Format`` of element ``i``.
        """
        return hipArray_Format(self._ptr[i].Format)
    def set_Format(self, i, value):
        """Setter for ``Format`` of element ``i``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].Format = value.value
    @property
    def Format(self):
        """Getter for ``Format``."""
        return self.get_Format(0)
    @Format.setter
    def Format(self, value):
        """Setter for ``Format``."""
        self.set_Format(0,value)
    def get_NumChannels(self, i):
        """Getter for ``NumChannels`` of element ``i``.
        """
        return self._ptr[i].NumChannels
    def set_NumChannels(self, i, unsigned int value):
        """Setter for ``NumChannels`` of element ``i``.
        """
        self._ptr[i].NumChannels = value
    @property
    def NumChannels(self):
        """Getter for ``NumChannels``."""
        return self.get_NumChannels(0)
    @NumChannels.setter
    def NumChannels(self, unsigned int value):
        """Setter for ``NumChannels``."""
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
        """Getter for ``Width`` of element ``i``.
        """
        return self._ptr[i].Width
    def set_Width(self, i, int value):
        """Setter for ``Width`` of element ``i``.
        """
        self._ptr[i].Width = value
    @property
    def Width(self):
        """Getter for ``Width``."""
        return self.get_Width(0)
    @Width.setter
    def Width(self, int value):
        """Setter for ``Width``."""
        self.set_Width(0,value)
    def get_Height(self, i):
        """Getter for ``Height`` of element ``i``.
        """
        return self._ptr[i].Height
    def set_Height(self, i, int value):
        """Setter for ``Height`` of element ``i``.
        """
        self._ptr[i].Height = value
    @property
    def Height(self):
        """Getter for ``Height``."""
        return self.get_Height(0)
    @Height.setter
    def Height(self, int value):
        """Setter for ``Height``."""
        self.set_Height(0,value)
    def get_Depth(self, i):
        """Getter for ``Depth`` of element ``i``.
        """
        return self._ptr[i].Depth
    def set_Depth(self, i, int value):
        """Setter for ``Depth`` of element ``i``.
        """
        self._ptr[i].Depth = value
    @property
    def Depth(self):
        """Getter for ``Depth``."""
        return self.get_Depth(0)
    @Depth.setter
    def Depth(self, int value):
        """Setter for ``Depth``."""
        self.set_Depth(0,value)
    def get_Format(self, i):
        """Getter for ``Format`` of element ``i``.
        """
        return hipArray_Format(self._ptr[i].Format)
    def set_Format(self, i, value):
        """Setter for ``Format`` of element ``i``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].Format = value.value
    @property
    def Format(self):
        """Getter for ``Format``."""
        return self.get_Format(0)
    @Format.setter
    def Format(self, value):
        """Setter for ``Format``."""
        self.set_Format(0,value)
    def get_NumChannels(self, i):
        """Getter for ``NumChannels`` of element ``i``.
        """
        return self._ptr[i].NumChannels
    def set_NumChannels(self, i, unsigned int value):
        """Setter for ``NumChannels`` of element ``i``.
        """
        self._ptr[i].NumChannels = value
    @property
    def NumChannels(self):
        """Getter for ``NumChannels``."""
        return self.get_NumChannels(0)
    @NumChannels.setter
    def NumChannels(self, unsigned int value):
        """Setter for ``NumChannels``."""
        self.set_NumChannels(0,value)
    def get_Flags(self, i):
        """Getter for ``Flags`` of element ``i``.
        """
        return self._ptr[i].Flags
    def set_Flags(self, i, unsigned int value):
        """Setter for ``Flags`` of element ``i``.
        """
        self._ptr[i].Flags = value
    @property
    def Flags(self):
        """Getter for ``Flags``."""
        return self.get_Flags(0)
    @Flags.setter
    def Flags(self, unsigned int value):
        """Setter for ``Flags``."""
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
        """Getter for ``desc`` of element ``i``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].desc)
    @property
    def desc(self):
        """Getter for ``desc``."""
        return self.get_desc(0)
    def get_type(self, i):
        """Getter for ``type`` of element ``i``.
        """
        return self._ptr[i].type
    def set_type(self, i, unsigned int value):
        """Setter for ``type`` of element ``i``.
        """
        self._ptr[i].type = value
    @property
    def type(self):
        """Getter for ``type``."""
        return self.get_type(0)
    @type.setter
    def type(self, unsigned int value):
        """Setter for ``type``."""
        self.set_type(0,value)
    def get_width(self, i):
        """Getter for ``width`` of element ``i``.
        """
        return self._ptr[i].width
    def set_width(self, i, unsigned int value):
        """Setter for ``width`` of element ``i``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        """Getter for ``width``."""
        return self.get_width(0)
    @width.setter
    def width(self, unsigned int value):
        """Setter for ``width``."""
        self.set_width(0,value)
    def get_height(self, i):
        """Getter for ``height`` of element ``i``.
        """
        return self._ptr[i].height
    def set_height(self, i, unsigned int value):
        """Setter for ``height`` of element ``i``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        """Getter for ``height``."""
        return self.get_height(0)
    @height.setter
    def height(self, unsigned int value):
        """Setter for ``height``."""
        self.set_height(0,value)
    def get_depth(self, i):
        """Getter for ``depth`` of element ``i``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, unsigned int value):
        """Setter for ``depth`` of element ``i``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        """Getter for ``depth``."""
        return self.get_depth(0)
    @depth.setter
    def depth(self, unsigned int value):
        """Setter for ``depth``."""
        self.set_depth(0,value)
    def get_Format(self, i):
        """Getter for ``Format`` of element ``i``.
        """
        return hipArray_Format(self._ptr[i].Format)
    def set_Format(self, i, value):
        """Setter for ``Format`` of element ``i``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].Format = value.value
    @property
    def Format(self):
        """Getter for ``Format``."""
        return self.get_Format(0)
    @Format.setter
    def Format(self, value):
        """Setter for ``Format``."""
        self.set_Format(0,value)
    def get_NumChannels(self, i):
        """Getter for ``NumChannels`` of element ``i``.
        """
        return self._ptr[i].NumChannels
    def set_NumChannels(self, i, unsigned int value):
        """Setter for ``NumChannels`` of element ``i``.
        """
        self._ptr[i].NumChannels = value
    @property
    def NumChannels(self):
        """Getter for ``NumChannels``."""
        return self.get_NumChannels(0)
    @NumChannels.setter
    def NumChannels(self, unsigned int value):
        """Setter for ``NumChannels``."""
        self.set_NumChannels(0,value)
    def get_isDrv(self, i):
        """Getter for ``isDrv`` of element ``i``.
        """
        return self._ptr[i].isDrv
    def set_isDrv(self, i, int value):
        """Setter for ``isDrv`` of element ``i``.
        """
        self._ptr[i].isDrv = value
    @property
    def isDrv(self):
        """Getter for ``isDrv``."""
        return self.get_isDrv(0)
    @isDrv.setter
    def isDrv(self, int value):
        """Setter for ``isDrv``."""
        self.set_isDrv(0,value)
    def get_textureType(self, i):
        """Getter for ``textureType`` of element ``i``.
        """
        return self._ptr[i].textureType
    def set_textureType(self, i, unsigned int value):
        """Setter for ``textureType`` of element ``i``.
        """
        self._ptr[i].textureType = value
    @property
    def textureType(self):
        """Getter for ``textureType``."""
        return self.get_textureType(0)
    @textureType.setter
    def textureType(self, unsigned int value):
        """Setter for ``textureType``."""
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
        """Getter for ``srcXInBytes`` of element ``i``.
        """
        return self._ptr[i].srcXInBytes
    def set_srcXInBytes(self, i, int value):
        """Setter for ``srcXInBytes`` of element ``i``.
        """
        self._ptr[i].srcXInBytes = value
    @property
    def srcXInBytes(self):
        """Getter for ``srcXInBytes``."""
        return self.get_srcXInBytes(0)
    @srcXInBytes.setter
    def srcXInBytes(self, int value):
        """Setter for ``srcXInBytes``."""
        self.set_srcXInBytes(0,value)
    def get_srcY(self, i):
        """Getter for ``srcY`` of element ``i``.
        """
        return self._ptr[i].srcY
    def set_srcY(self, i, int value):
        """Setter for ``srcY`` of element ``i``.
        """
        self._ptr[i].srcY = value
    @property
    def srcY(self):
        """Getter for ``srcY``."""
        return self.get_srcY(0)
    @srcY.setter
    def srcY(self, int value):
        """Setter for ``srcY``."""
        self.set_srcY(0,value)
    def get_srcMemoryType(self, i):
        """Getter for ``srcMemoryType`` of element ``i``.
        """
        return hipMemoryType(self._ptr[i].srcMemoryType)
    def set_srcMemoryType(self, i, value):
        """Setter for ``srcMemoryType`` of element ``i``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].srcMemoryType = value.value
    @property
    def srcMemoryType(self):
        """Getter for ``srcMemoryType``."""
        return self.get_srcMemoryType(0)
    @srcMemoryType.setter
    def srcMemoryType(self, value):
        """Setter for ``srcMemoryType``."""
        self.set_srcMemoryType(0,value)
    def get_srcPitch(self, i):
        """Getter for ``srcPitch`` of element ``i``.
        """
        return self._ptr[i].srcPitch
    def set_srcPitch(self, i, int value):
        """Setter for ``srcPitch`` of element ``i``.
        """
        self._ptr[i].srcPitch = value
    @property
    def srcPitch(self):
        """Getter for ``srcPitch``."""
        return self.get_srcPitch(0)
    @srcPitch.setter
    def srcPitch(self, int value):
        """Setter for ``srcPitch``."""
        self.set_srcPitch(0,value)
    def get_dstXInBytes(self, i):
        """Getter for ``dstXInBytes`` of element ``i``.
        """
        return self._ptr[i].dstXInBytes
    def set_dstXInBytes(self, i, int value):
        """Setter for ``dstXInBytes`` of element ``i``.
        """
        self._ptr[i].dstXInBytes = value
    @property
    def dstXInBytes(self):
        """Getter for ``dstXInBytes``."""
        return self.get_dstXInBytes(0)
    @dstXInBytes.setter
    def dstXInBytes(self, int value):
        """Setter for ``dstXInBytes``."""
        self.set_dstXInBytes(0,value)
    def get_dstY(self, i):
        """Getter for ``dstY`` of element ``i``.
        """
        return self._ptr[i].dstY
    def set_dstY(self, i, int value):
        """Setter for ``dstY`` of element ``i``.
        """
        self._ptr[i].dstY = value
    @property
    def dstY(self):
        """Getter for ``dstY``."""
        return self.get_dstY(0)
    @dstY.setter
    def dstY(self, int value):
        """Setter for ``dstY``."""
        self.set_dstY(0,value)
    def get_dstMemoryType(self, i):
        """Getter for ``dstMemoryType`` of element ``i``.
        """
        return hipMemoryType(self._ptr[i].dstMemoryType)
    def set_dstMemoryType(self, i, value):
        """Setter for ``dstMemoryType`` of element ``i``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].dstMemoryType = value.value
    @property
    def dstMemoryType(self):
        """Getter for ``dstMemoryType``."""
        return self.get_dstMemoryType(0)
    @dstMemoryType.setter
    def dstMemoryType(self, value):
        """Setter for ``dstMemoryType``."""
        self.set_dstMemoryType(0,value)
    def get_dstPitch(self, i):
        """Getter for ``dstPitch`` of element ``i``.
        """
        return self._ptr[i].dstPitch
    def set_dstPitch(self, i, int value):
        """Setter for ``dstPitch`` of element ``i``.
        """
        self._ptr[i].dstPitch = value
    @property
    def dstPitch(self):
        """Getter for ``dstPitch``."""
        return self.get_dstPitch(0)
    @dstPitch.setter
    def dstPitch(self, int value):
        """Setter for ``dstPitch``."""
        self.set_dstPitch(0,value)
    def get_WidthInBytes(self, i):
        """Getter for ``WidthInBytes`` of element ``i``.
        """
        return self._ptr[i].WidthInBytes
    def set_WidthInBytes(self, i, int value):
        """Setter for ``WidthInBytes`` of element ``i``.
        """
        self._ptr[i].WidthInBytes = value
    @property
    def WidthInBytes(self):
        """Getter for ``WidthInBytes``."""
        return self.get_WidthInBytes(0)
    @WidthInBytes.setter
    def WidthInBytes(self, int value):
        """Setter for ``WidthInBytes``."""
        self.set_WidthInBytes(0,value)
    def get_Height(self, i):
        """Getter for ``Height`` of element ``i``.
        """
        return self._ptr[i].Height
    def set_Height(self, i, int value):
        """Setter for ``Height`` of element ``i``.
        """
        self._ptr[i].Height = value
    @property
    def Height(self):
        """Getter for ``Height``."""
        return self.get_Height(0)
    @Height.setter
    def Height(self, int value):
        """Setter for ``Height``."""
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
        """Getter for ``desc`` of element ``i``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].desc)
    @property
    def desc(self):
        """Getter for ``desc``."""
        return self.get_desc(0)
    def get_type(self, i):
        """Getter for ``type`` of element ``i``.
        """
        return self._ptr[i].type
    def set_type(self, i, unsigned int value):
        """Setter for ``type`` of element ``i``.
        """
        self._ptr[i].type = value
    @property
    def type(self):
        """Getter for ``type``."""
        return self.get_type(0)
    @type.setter
    def type(self, unsigned int value):
        """Setter for ``type``."""
        self.set_type(0,value)
    def get_width(self, i):
        """Getter for ``width`` of element ``i``.
        """
        return self._ptr[i].width
    def set_width(self, i, unsigned int value):
        """Setter for ``width`` of element ``i``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        """Getter for ``width``."""
        return self.get_width(0)
    @width.setter
    def width(self, unsigned int value):
        """Setter for ``width``."""
        self.set_width(0,value)
    def get_height(self, i):
        """Getter for ``height`` of element ``i``.
        """
        return self._ptr[i].height
    def set_height(self, i, unsigned int value):
        """Setter for ``height`` of element ``i``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        """Getter for ``height``."""
        return self.get_height(0)
    @height.setter
    def height(self, unsigned int value):
        """Setter for ``height``."""
        self.set_height(0,value)
    def get_depth(self, i):
        """Getter for ``depth`` of element ``i``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, unsigned int value):
        """Setter for ``depth`` of element ``i``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        """Getter for ``depth``."""
        return self.get_depth(0)
    @depth.setter
    def depth(self, unsigned int value):
        """Setter for ``depth``."""
        self.set_depth(0,value)
    def get_min_mipmap_level(self, i):
        """Getter for ``min_mipmap_level`` of element ``i``.
        """
        return self._ptr[i].min_mipmap_level
    def set_min_mipmap_level(self, i, unsigned int value):
        """Setter for ``min_mipmap_level`` of element ``i``.
        """
        self._ptr[i].min_mipmap_level = value
    @property
    def min_mipmap_level(self):
        """Getter for ``min_mipmap_level``."""
        return self.get_min_mipmap_level(0)
    @min_mipmap_level.setter
    def min_mipmap_level(self, unsigned int value):
        """Setter for ``min_mipmap_level``."""
        self.set_min_mipmap_level(0,value)
    def get_max_mipmap_level(self, i):
        """Getter for ``max_mipmap_level`` of element ``i``.
        """
        return self._ptr[i].max_mipmap_level
    def set_max_mipmap_level(self, i, unsigned int value):
        """Setter for ``max_mipmap_level`` of element ``i``.
        """
        self._ptr[i].max_mipmap_level = value
    @property
    def max_mipmap_level(self):
        """Getter for ``max_mipmap_level``."""
        return self.get_max_mipmap_level(0)
    @max_mipmap_level.setter
    def max_mipmap_level(self, unsigned int value):
        """Setter for ``max_mipmap_level``."""
        self.set_max_mipmap_level(0,value)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Setter for ``flags`` of element ``i``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        """Setter for ``flags``."""
        self.set_flags(0,value)
    def get_format(self, i):
        """Getter for ``format`` of element ``i``.
        """
        return hipArray_Format(self._ptr[i].format)
    def set_format(self, i, value):
        """Setter for ``format`` of element ``i``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        """Getter for ``format``."""
        return self.get_format(0)
    @format.setter
    def format(self, value):
        """Setter for ``format``."""
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
    def get_filterMode(self, i):
        """Getter for ``filterMode`` of element ``i``.
        """
        return HIPfilter_mode_enum(self._ptr[i].filterMode)
    def set_filterMode(self, i, value):
        """Setter for ``filterMode`` of element ``i``.
        """
        if not isinstance(value, HIPfilter_mode_enum):
            raise TypeError("'value' must be of type 'HIPfilter_mode_enum'")
        self._ptr[i].filterMode = value.value
    @property
    def filterMode(self):
        """Getter for ``filterMode``."""
        return self.get_filterMode(0)
    @filterMode.setter
    def filterMode(self, value):
        """Setter for ``filterMode``."""
        self.set_filterMode(0,value)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Setter for ``flags`` of element ``i``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        """Setter for ``flags``."""
        self.set_flags(0,value)
    def get_maxAnisotropy(self, i):
        """Getter for ``maxAnisotropy`` of element ``i``.
        """
        return self._ptr[i].maxAnisotropy
    def set_maxAnisotropy(self, i, unsigned int value):
        """Setter for ``maxAnisotropy`` of element ``i``.
        """
        self._ptr[i].maxAnisotropy = value
    @property
    def maxAnisotropy(self):
        """Getter for ``maxAnisotropy``."""
        return self.get_maxAnisotropy(0)
    @maxAnisotropy.setter
    def maxAnisotropy(self, unsigned int value):
        """Setter for ``maxAnisotropy``."""
        self.set_maxAnisotropy(0,value)
    def get_mipmapFilterMode(self, i):
        """Getter for ``mipmapFilterMode`` of element ``i``.
        """
        return HIPfilter_mode_enum(self._ptr[i].mipmapFilterMode)
    def set_mipmapFilterMode(self, i, value):
        """Setter for ``mipmapFilterMode`` of element ``i``.
        """
        if not isinstance(value, HIPfilter_mode_enum):
            raise TypeError("'value' must be of type 'HIPfilter_mode_enum'")
        self._ptr[i].mipmapFilterMode = value.value
    @property
    def mipmapFilterMode(self):
        """Getter for ``mipmapFilterMode``."""
        return self.get_mipmapFilterMode(0)
    @mipmapFilterMode.setter
    def mipmapFilterMode(self, value):
        """Setter for ``mipmapFilterMode``."""
        self.set_mipmapFilterMode(0,value)
    def get_mipmapLevelBias(self, i):
        """Getter for ``mipmapLevelBias`` of element ``i``.
        """
        return self._ptr[i].mipmapLevelBias
    def set_mipmapLevelBias(self, i, float value):
        """Setter for ``mipmapLevelBias`` of element ``i``.
        """
        self._ptr[i].mipmapLevelBias = value
    @property
    def mipmapLevelBias(self):
        """Getter for ``mipmapLevelBias``."""
        return self.get_mipmapLevelBias(0)
    @mipmapLevelBias.setter
    def mipmapLevelBias(self, float value):
        """Setter for ``mipmapLevelBias``."""
        self.set_mipmapLevelBias(0,value)
    def get_minMipmapLevelClamp(self, i):
        """Getter for ``minMipmapLevelClamp`` of element ``i``.
        """
        return self._ptr[i].minMipmapLevelClamp
    def set_minMipmapLevelClamp(self, i, float value):
        """Setter for ``minMipmapLevelClamp`` of element ``i``.
        """
        self._ptr[i].minMipmapLevelClamp = value
    @property
    def minMipmapLevelClamp(self):
        """Getter for ``minMipmapLevelClamp``."""
        return self.get_minMipmapLevelClamp(0)
    @minMipmapLevelClamp.setter
    def minMipmapLevelClamp(self, float value):
        """Setter for ``minMipmapLevelClamp``."""
        self.set_minMipmapLevelClamp(0,value)
    def get_maxMipmapLevelClamp(self, i):
        """Getter for ``maxMipmapLevelClamp`` of element ``i``.
        """
        return self._ptr[i].maxMipmapLevelClamp
    def set_maxMipmapLevelClamp(self, i, float value):
        """Setter for ``maxMipmapLevelClamp`` of element ``i``.
        """
        self._ptr[i].maxMipmapLevelClamp = value
    @property
    def maxMipmapLevelClamp(self):
        """Getter for ``maxMipmapLevelClamp``."""
        return self.get_maxMipmapLevelClamp(0)
    @maxMipmapLevelClamp.setter
    def maxMipmapLevelClamp(self, float value):
        """Setter for ``maxMipmapLevelClamp``."""
        self.set_maxMipmapLevelClamp(0,value)


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
    def get_desc(self, i):
        """Getter for ``desc`` of element ``i``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].desc)
    @property
    def desc(self):
        """Getter for ``desc``."""
        return self.get_desc(0)
    def get_sizeInBytes(self, i):
        """Getter for ``sizeInBytes`` of element ``i``.
        """
        return self._ptr[i].sizeInBytes
    def set_sizeInBytes(self, i, int value):
        """Setter for ``sizeInBytes`` of element ``i``.
        """
        self._ptr[i].sizeInBytes = value
    @property
    def sizeInBytes(self):
        """Getter for ``sizeInBytes``."""
        return self.get_sizeInBytes(0)
    @sizeInBytes.setter
    def sizeInBytes(self, int value):
        """Setter for ``sizeInBytes``."""
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
        """Getter for ``desc`` of element ``i``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].desc)
    @property
    def desc(self):
        """Getter for ``desc``."""
        return self.get_desc(0)
    def get_width(self, i):
        """Getter for ``width`` of element ``i``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Setter for ``width`` of element ``i``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        """Getter for ``width``."""
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        """Setter for ``width``."""
        self.set_width(0,value)
    def get_height(self, i):
        """Getter for ``height`` of element ``i``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Setter for ``height`` of element ``i``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        """Getter for ``height``."""
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        """Setter for ``height``."""
        self.set_height(0,value)
    def get_pitchInBytes(self, i):
        """Getter for ``pitchInBytes`` of element ``i``.
        """
        return self._ptr[i].pitchInBytes
    def set_pitchInBytes(self, i, int value):
        """Setter for ``pitchInBytes`` of element ``i``.
        """
        self._ptr[i].pitchInBytes = value
    @property
    def pitchInBytes(self):
        """Getter for ``pitchInBytes``."""
        return self.get_pitchInBytes(0)
    @pitchInBytes.setter
    def pitchInBytes(self, int value):
        """Setter for ``pitchInBytes``."""
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
        """Getter for ``array`` of element ``i``.
        """
        return hipResourceDesc_union_0_struct_0.from_ptr(&self._ptr[i].array)
    @property
    def array(self):
        """Getter for ``array``."""
        return self.get_array(0)
    def get_mipmap(self, i):
        """Getter for ``mipmap`` of element ``i``.
        """
        return hipResourceDesc_union_0_struct_1.from_ptr(&self._ptr[i].mipmap)
    @property
    def mipmap(self):
        """Getter for ``mipmap``."""
        return self.get_mipmap(0)
    def get_linear(self, i):
        """Getter for ``linear`` of element ``i``.
        """
        return hipResourceDesc_union_0_struct_2.from_ptr(&self._ptr[i].linear)
    @property
    def linear(self):
        """Getter for ``linear``."""
        return self.get_linear(0)
    def get_pitch2D(self, i):
        """Getter for ``pitch2D`` of element ``i``.
        """
        return hipResourceDesc_union_0_struct_3.from_ptr(&self._ptr[i].pitch2D)
    @property
    def pitch2D(self):
        """Getter for ``pitch2D``."""
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
        """Getter for ``resType`` of element ``i``.
        """
        return hipResourceType(self._ptr[i].resType)
    def set_resType(self, i, value):
        """Setter for ``resType`` of element ``i``.
        """
        if not isinstance(value, hipResourceType):
            raise TypeError("'value' must be of type 'hipResourceType'")
        self._ptr[i].resType = value.value
    @property
    def resType(self):
        """Getter for ``resType``."""
        return self.get_resType(0)
    @resType.setter
    def resType(self, value):
        """Setter for ``resType``."""
        self.set_resType(0,value)
    def get_res(self, i):
        """Getter for ``res`` of element ``i``.
        """
        return hipResourceDesc_union_0.from_ptr(&self._ptr[i].res)
    @property
    def res(self):
        """Getter for ``res``."""
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
        """Getter for ``format`` of element ``i``.
        """
        return hipArray_Format(self._ptr[i].format)
    def set_format(self, i, value):
        """Setter for ``format`` of element ``i``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        """Getter for ``format``."""
        return self.get_format(0)
    @format.setter
    def format(self, value):
        """Setter for ``format``."""
        self.set_format(0,value)
    def get_numChannels(self, i):
        """Getter for ``numChannels`` of element ``i``.
        """
        return self._ptr[i].numChannels
    def set_numChannels(self, i, unsigned int value):
        """Setter for ``numChannels`` of element ``i``.
        """
        self._ptr[i].numChannels = value
    @property
    def numChannels(self):
        """Getter for ``numChannels``."""
        return self.get_numChannels(0)
    @numChannels.setter
    def numChannels(self, unsigned int value):
        """Setter for ``numChannels``."""
        self.set_numChannels(0,value)
    def get_sizeInBytes(self, i):
        """Getter for ``sizeInBytes`` of element ``i``.
        """
        return self._ptr[i].sizeInBytes
    def set_sizeInBytes(self, i, int value):
        """Setter for ``sizeInBytes`` of element ``i``.
        """
        self._ptr[i].sizeInBytes = value
    @property
    def sizeInBytes(self):
        """Getter for ``sizeInBytes``."""
        return self.get_sizeInBytes(0)
    @sizeInBytes.setter
    def sizeInBytes(self, int value):
        """Setter for ``sizeInBytes``."""
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
        """Getter for ``format`` of element ``i``.
        """
        return hipArray_Format(self._ptr[i].format)
    def set_format(self, i, value):
        """Setter for ``format`` of element ``i``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        """Getter for ``format``."""
        return self.get_format(0)
    @format.setter
    def format(self, value):
        """Setter for ``format``."""
        self.set_format(0,value)
    def get_numChannels(self, i):
        """Getter for ``numChannels`` of element ``i``.
        """
        return self._ptr[i].numChannels
    def set_numChannels(self, i, unsigned int value):
        """Setter for ``numChannels`` of element ``i``.
        """
        self._ptr[i].numChannels = value
    @property
    def numChannels(self):
        """Getter for ``numChannels``."""
        return self.get_numChannels(0)
    @numChannels.setter
    def numChannels(self, unsigned int value):
        """Setter for ``numChannels``."""
        self.set_numChannels(0,value)
    def get_width(self, i):
        """Getter for ``width`` of element ``i``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Setter for ``width`` of element ``i``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        """Getter for ``width``."""
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        """Setter for ``width``."""
        self.set_width(0,value)
    def get_height(self, i):
        """Getter for ``height`` of element ``i``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Setter for ``height`` of element ``i``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        """Getter for ``height``."""
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        """Setter for ``height``."""
        self.set_height(0,value)
    def get_pitchInBytes(self, i):
        """Getter for ``pitchInBytes`` of element ``i``.
        """
        return self._ptr[i].pitchInBytes
    def set_pitchInBytes(self, i, int value):
        """Setter for ``pitchInBytes`` of element ``i``.
        """
        self._ptr[i].pitchInBytes = value
    @property
    def pitchInBytes(self):
        """Getter for ``pitchInBytes``."""
        return self.get_pitchInBytes(0)
    @pitchInBytes.setter
    def pitchInBytes(self, int value):
        """Setter for ``pitchInBytes``."""
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
        """Getter for ``array`` of element ``i``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_0.from_ptr(&self._ptr[i].array)
    @property
    def array(self):
        """Getter for ``array``."""
        return self.get_array(0)
    def get_mipmap(self, i):
        """Getter for ``mipmap`` of element ``i``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_1.from_ptr(&self._ptr[i].mipmap)
    @property
    def mipmap(self):
        """Getter for ``mipmap``."""
        return self.get_mipmap(0)
    def get_linear(self, i):
        """Getter for ``linear`` of element ``i``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_2.from_ptr(&self._ptr[i].linear)
    @property
    def linear(self):
        """Getter for ``linear``."""
        return self.get_linear(0)
    def get_pitch2D(self, i):
        """Getter for ``pitch2D`` of element ``i``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_3.from_ptr(&self._ptr[i].pitch2D)
    @property
    def pitch2D(self):
        """Getter for ``pitch2D``."""
        return self.get_pitch2D(0)
    def get_reserved(self, i):
        """Getter for ``reserved`` of element ``i``.
        """
        return HIP_RESOURCE_DESC_st_union_0_struct_4.from_ptr(&self._ptr[i].reserved)
    @property
    def reserved(self):
        """Getter for ``reserved``."""
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
        """Getter for ``resType`` of element ``i``.
        """
        return HIPresourcetype_enum(self._ptr[i].resType)
    def set_resType(self, i, value):
        """Setter for ``resType`` of element ``i``.
        """
        if not isinstance(value, HIPresourcetype_enum):
            raise TypeError("'value' must be of type 'HIPresourcetype_enum'")
        self._ptr[i].resType = value.value
    @property
    def resType(self):
        """Getter for ``resType``."""
        return self.get_resType(0)
    @resType.setter
    def resType(self, value):
        """Setter for ``resType``."""
        self.set_resType(0,value)
    def get_res(self, i):
        """Getter for ``res`` of element ``i``.
        """
        return HIP_RESOURCE_DESC_st_union_0.from_ptr(&self._ptr[i].res)
    @property
    def res(self):
        """Getter for ``res``."""
        return self.get_res(0)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Setter for ``flags`` of element ``i``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        """Setter for ``flags``."""
        self.set_flags(0,value)



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
        """Getter for ``format`` of element ``i``.
        """
        return hipResourceViewFormat(self._ptr[i].format)
    def set_format(self, i, value):
        """Setter for ``format`` of element ``i``.
        """
        if not isinstance(value, hipResourceViewFormat):
            raise TypeError("'value' must be of type 'hipResourceViewFormat'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        """Getter for ``format``."""
        return self.get_format(0)
    @format.setter
    def format(self, value):
        """Setter for ``format``."""
        self.set_format(0,value)
    def get_width(self, i):
        """Getter for ``width`` of element ``i``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Setter for ``width`` of element ``i``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        """Getter for ``width``."""
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        """Setter for ``width``."""
        self.set_width(0,value)
    def get_height(self, i):
        """Getter for ``height`` of element ``i``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Setter for ``height`` of element ``i``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        """Getter for ``height``."""
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        """Setter for ``height``."""
        self.set_height(0,value)
    def get_depth(self, i):
        """Getter for ``depth`` of element ``i``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, int value):
        """Setter for ``depth`` of element ``i``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        """Getter for ``depth``."""
        return self.get_depth(0)
    @depth.setter
    def depth(self, int value):
        """Setter for ``depth``."""
        self.set_depth(0,value)
    def get_firstMipmapLevel(self, i):
        """Getter for ``firstMipmapLevel`` of element ``i``.
        """
        return self._ptr[i].firstMipmapLevel
    def set_firstMipmapLevel(self, i, unsigned int value):
        """Setter for ``firstMipmapLevel`` of element ``i``.
        """
        self._ptr[i].firstMipmapLevel = value
    @property
    def firstMipmapLevel(self):
        """Getter for ``firstMipmapLevel``."""
        return self.get_firstMipmapLevel(0)
    @firstMipmapLevel.setter
    def firstMipmapLevel(self, unsigned int value):
        """Setter for ``firstMipmapLevel``."""
        self.set_firstMipmapLevel(0,value)
    def get_lastMipmapLevel(self, i):
        """Getter for ``lastMipmapLevel`` of element ``i``.
        """
        return self._ptr[i].lastMipmapLevel
    def set_lastMipmapLevel(self, i, unsigned int value):
        """Setter for ``lastMipmapLevel`` of element ``i``.
        """
        self._ptr[i].lastMipmapLevel = value
    @property
    def lastMipmapLevel(self):
        """Getter for ``lastMipmapLevel``."""
        return self.get_lastMipmapLevel(0)
    @lastMipmapLevel.setter
    def lastMipmapLevel(self, unsigned int value):
        """Setter for ``lastMipmapLevel``."""
        self.set_lastMipmapLevel(0,value)
    def get_firstLayer(self, i):
        """Getter for ``firstLayer`` of element ``i``.
        """
        return self._ptr[i].firstLayer
    def set_firstLayer(self, i, unsigned int value):
        """Setter for ``firstLayer`` of element ``i``.
        """
        self._ptr[i].firstLayer = value
    @property
    def firstLayer(self):
        """Getter for ``firstLayer``."""
        return self.get_firstLayer(0)
    @firstLayer.setter
    def firstLayer(self, unsigned int value):
        """Setter for ``firstLayer``."""
        self.set_firstLayer(0,value)
    def get_lastLayer(self, i):
        """Getter for ``lastLayer`` of element ``i``.
        """
        return self._ptr[i].lastLayer
    def set_lastLayer(self, i, unsigned int value):
        """Setter for ``lastLayer`` of element ``i``.
        """
        self._ptr[i].lastLayer = value
    @property
    def lastLayer(self):
        """Getter for ``lastLayer``."""
        return self.get_lastLayer(0)
    @lastLayer.setter
    def lastLayer(self, unsigned int value):
        """Setter for ``lastLayer``."""
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
        """Getter for ``format`` of element ``i``.
        """
        return HIPresourceViewFormat_enum(self._ptr[i].format)
    def set_format(self, i, value):
        """Setter for ``format`` of element ``i``.
        """
        if not isinstance(value, HIPresourceViewFormat_enum):
            raise TypeError("'value' must be of type 'HIPresourceViewFormat_enum'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        """Getter for ``format``."""
        return self.get_format(0)
    @format.setter
    def format(self, value):
        """Setter for ``format``."""
        self.set_format(0,value)
    def get_width(self, i):
        """Getter for ``width`` of element ``i``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Setter for ``width`` of element ``i``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        """Getter for ``width``."""
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        """Setter for ``width``."""
        self.set_width(0,value)
    def get_height(self, i):
        """Getter for ``height`` of element ``i``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Setter for ``height`` of element ``i``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        """Getter for ``height``."""
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        """Setter for ``height``."""
        self.set_height(0,value)
    def get_depth(self, i):
        """Getter for ``depth`` of element ``i``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, int value):
        """Setter for ``depth`` of element ``i``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        """Getter for ``depth``."""
        return self.get_depth(0)
    @depth.setter
    def depth(self, int value):
        """Setter for ``depth``."""
        self.set_depth(0,value)
    def get_firstMipmapLevel(self, i):
        """Getter for ``firstMipmapLevel`` of element ``i``.
        """
        return self._ptr[i].firstMipmapLevel
    def set_firstMipmapLevel(self, i, unsigned int value):
        """Setter for ``firstMipmapLevel`` of element ``i``.
        """
        self._ptr[i].firstMipmapLevel = value
    @property
    def firstMipmapLevel(self):
        """Getter for ``firstMipmapLevel``."""
        return self.get_firstMipmapLevel(0)
    @firstMipmapLevel.setter
    def firstMipmapLevel(self, unsigned int value):
        """Setter for ``firstMipmapLevel``."""
        self.set_firstMipmapLevel(0,value)
    def get_lastMipmapLevel(self, i):
        """Getter for ``lastMipmapLevel`` of element ``i``.
        """
        return self._ptr[i].lastMipmapLevel
    def set_lastMipmapLevel(self, i, unsigned int value):
        """Setter for ``lastMipmapLevel`` of element ``i``.
        """
        self._ptr[i].lastMipmapLevel = value
    @property
    def lastMipmapLevel(self):
        """Getter for ``lastMipmapLevel``."""
        return self.get_lastMipmapLevel(0)
    @lastMipmapLevel.setter
    def lastMipmapLevel(self, unsigned int value):
        """Setter for ``lastMipmapLevel``."""
        self.set_lastMipmapLevel(0,value)
    def get_firstLayer(self, i):
        """Getter for ``firstLayer`` of element ``i``.
        """
        return self._ptr[i].firstLayer
    def set_firstLayer(self, i, unsigned int value):
        """Setter for ``firstLayer`` of element ``i``.
        """
        self._ptr[i].firstLayer = value
    @property
    def firstLayer(self):
        """Getter for ``firstLayer``."""
        return self.get_firstLayer(0)
    @firstLayer.setter
    def firstLayer(self, unsigned int value):
        """Setter for ``firstLayer``."""
        self.set_firstLayer(0,value)
    def get_lastLayer(self, i):
        """Getter for ``lastLayer`` of element ``i``.
        """
        return self._ptr[i].lastLayer
    def set_lastLayer(self, i, unsigned int value):
        """Setter for ``lastLayer`` of element ``i``.
        """
        self._ptr[i].lastLayer = value
    @property
    def lastLayer(self):
        """Getter for ``lastLayer``."""
        return self.get_lastLayer(0)
    @lastLayer.setter
    def lastLayer(self, unsigned int value):
        """Setter for ``lastLayer``."""
        self.set_lastLayer(0,value)


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
        """Getter for ``pitch`` of element ``i``.
        """
        return self._ptr[i].pitch
    def set_pitch(self, i, int value):
        """Setter for ``pitch`` of element ``i``.
        """
        self._ptr[i].pitch = value
    @property
    def pitch(self):
        """Getter for ``pitch``."""
        return self.get_pitch(0)
    @pitch.setter
    def pitch(self, int value):
        """Setter for ``pitch``."""
        self.set_pitch(0,value)
    def get_xsize(self, i):
        """Getter for ``xsize`` of element ``i``.
        """
        return self._ptr[i].xsize
    def set_xsize(self, i, int value):
        """Setter for ``xsize`` of element ``i``.
        """
        self._ptr[i].xsize = value
    @property
    def xsize(self):
        """Getter for ``xsize``."""
        return self.get_xsize(0)
    @xsize.setter
    def xsize(self, int value):
        """Setter for ``xsize``."""
        self.set_xsize(0,value)
    def get_ysize(self, i):
        """Getter for ``ysize`` of element ``i``.
        """
        return self._ptr[i].ysize
    def set_ysize(self, i, int value):
        """Setter for ``ysize`` of element ``i``.
        """
        self._ptr[i].ysize = value
    @property
    def ysize(self):
        """Getter for ``ysize``."""
        return self.get_ysize(0)
    @ysize.setter
    def ysize(self, int value):
        """Setter for ``ysize``."""
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
        """Getter for ``width`` of element ``i``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Setter for ``width`` of element ``i``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        """Getter for ``width``."""
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        """Setter for ``width``."""
        self.set_width(0,value)
    def get_height(self, i):
        """Getter for ``height`` of element ``i``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Setter for ``height`` of element ``i``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        """Getter for ``height``."""
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        """Setter for ``height``."""
        self.set_height(0,value)
    def get_depth(self, i):
        """Getter for ``depth`` of element ``i``.
        """
        return self._ptr[i].depth
    def set_depth(self, i, int value):
        """Setter for ``depth`` of element ``i``.
        """
        self._ptr[i].depth = value
    @property
    def depth(self):
        """Getter for ``depth``."""
        return self.get_depth(0)
    @depth.setter
    def depth(self, int value):
        """Setter for ``depth``."""
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
        """Getter for ``x`` of element ``i``.
        """
        return self._ptr[i].x
    def set_x(self, i, int value):
        """Setter for ``x`` of element ``i``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        """Getter for ``x``."""
        return self.get_x(0)
    @x.setter
    def x(self, int value):
        """Setter for ``x``."""
        self.set_x(0,value)
    def get_y(self, i):
        """Getter for ``y`` of element ``i``.
        """
        return self._ptr[i].y
    def set_y(self, i, int value):
        """Setter for ``y`` of element ``i``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        """Getter for ``y``."""
        return self.get_y(0)
    @y.setter
    def y(self, int value):
        """Setter for ``y``."""
        self.set_y(0,value)
    def get_z(self, i):
        """Getter for ``z`` of element ``i``.
        """
        return self._ptr[i].z
    def set_z(self, i, int value):
        """Setter for ``z`` of element ``i``.
        """
        self._ptr[i].z = value
    @property
    def z(self):
        """Getter for ``z``."""
        return self.get_z(0)
    @z.setter
    def z(self, int value):
        """Setter for ``z``."""
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
        """Getter for ``srcPos`` of element ``i``.
        """
        return hipPos.from_ptr(&self._ptr[i].srcPos)
    @property
    def srcPos(self):
        """Getter for ``srcPos``."""
        return self.get_srcPos(0)
    def get_srcPtr(self, i):
        """Getter for ``srcPtr`` of element ``i``.
        """
        return hipPitchedPtr.from_ptr(&self._ptr[i].srcPtr)
    @property
    def srcPtr(self):
        """Getter for ``srcPtr``."""
        return self.get_srcPtr(0)
    def get_dstPos(self, i):
        """Getter for ``dstPos`` of element ``i``.
        """
        return hipPos.from_ptr(&self._ptr[i].dstPos)
    @property
    def dstPos(self):
        """Getter for ``dstPos``."""
        return self.get_dstPos(0)
    def get_dstPtr(self, i):
        """Getter for ``dstPtr`` of element ``i``.
        """
        return hipPitchedPtr.from_ptr(&self._ptr[i].dstPtr)
    @property
    def dstPtr(self):
        """Getter for ``dstPtr``."""
        return self.get_dstPtr(0)
    def get_extent(self, i):
        """Getter for ``extent`` of element ``i``.
        """
        return hipExtent.from_ptr(&self._ptr[i].extent)
    @property
    def extent(self):
        """Getter for ``extent``."""
        return self.get_extent(0)
    def get_kind(self, i):
        """Getter for ``kind`` of element ``i``.
        """
        return hipMemcpyKind(self._ptr[i].kind)
    def set_kind(self, i, value):
        """Setter for ``kind`` of element ``i``.
        """
        if not isinstance(value, hipMemcpyKind):
            raise TypeError("'value' must be of type 'hipMemcpyKind'")
        self._ptr[i].kind = value.value
    @property
    def kind(self):
        """Getter for ``kind``."""
        return self.get_kind(0)
    @kind.setter
    def kind(self, value):
        """Setter for ``kind``."""
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
        """Getter for ``srcXInBytes`` of element ``i``.
        """
        return self._ptr[i].srcXInBytes
    def set_srcXInBytes(self, i, unsigned int value):
        """Setter for ``srcXInBytes`` of element ``i``.
        """
        self._ptr[i].srcXInBytes = value
    @property
    def srcXInBytes(self):
        """Getter for ``srcXInBytes``."""
        return self.get_srcXInBytes(0)
    @srcXInBytes.setter
    def srcXInBytes(self, unsigned int value):
        """Setter for ``srcXInBytes``."""
        self.set_srcXInBytes(0,value)
    def get_srcY(self, i):
        """Getter for ``srcY`` of element ``i``.
        """
        return self._ptr[i].srcY
    def set_srcY(self, i, unsigned int value):
        """Setter for ``srcY`` of element ``i``.
        """
        self._ptr[i].srcY = value
    @property
    def srcY(self):
        """Getter for ``srcY``."""
        return self.get_srcY(0)
    @srcY.setter
    def srcY(self, unsigned int value):
        """Setter for ``srcY``."""
        self.set_srcY(0,value)
    def get_srcZ(self, i):
        """Getter for ``srcZ`` of element ``i``.
        """
        return self._ptr[i].srcZ
    def set_srcZ(self, i, unsigned int value):
        """Setter for ``srcZ`` of element ``i``.
        """
        self._ptr[i].srcZ = value
    @property
    def srcZ(self):
        """Getter for ``srcZ``."""
        return self.get_srcZ(0)
    @srcZ.setter
    def srcZ(self, unsigned int value):
        """Setter for ``srcZ``."""
        self.set_srcZ(0,value)
    def get_srcLOD(self, i):
        """Getter for ``srcLOD`` of element ``i``.
        """
        return self._ptr[i].srcLOD
    def set_srcLOD(self, i, unsigned int value):
        """Setter for ``srcLOD`` of element ``i``.
        """
        self._ptr[i].srcLOD = value
    @property
    def srcLOD(self):
        """Getter for ``srcLOD``."""
        return self.get_srcLOD(0)
    @srcLOD.setter
    def srcLOD(self, unsigned int value):
        """Setter for ``srcLOD``."""
        self.set_srcLOD(0,value)
    def get_srcMemoryType(self, i):
        """Getter for ``srcMemoryType`` of element ``i``.
        """
        return hipMemoryType(self._ptr[i].srcMemoryType)
    def set_srcMemoryType(self, i, value):
        """Setter for ``srcMemoryType`` of element ``i``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].srcMemoryType = value.value
    @property
    def srcMemoryType(self):
        """Getter for ``srcMemoryType``."""
        return self.get_srcMemoryType(0)
    @srcMemoryType.setter
    def srcMemoryType(self, value):
        """Setter for ``srcMemoryType``."""
        self.set_srcMemoryType(0,value)
    def get_srcPitch(self, i):
        """Getter for ``srcPitch`` of element ``i``.
        """
        return self._ptr[i].srcPitch
    def set_srcPitch(self, i, unsigned int value):
        """Setter for ``srcPitch`` of element ``i``.
        """
        self._ptr[i].srcPitch = value
    @property
    def srcPitch(self):
        """Getter for ``srcPitch``."""
        return self.get_srcPitch(0)
    @srcPitch.setter
    def srcPitch(self, unsigned int value):
        """Setter for ``srcPitch``."""
        self.set_srcPitch(0,value)
    def get_srcHeight(self, i):
        """Getter for ``srcHeight`` of element ``i``.
        """
        return self._ptr[i].srcHeight
    def set_srcHeight(self, i, unsigned int value):
        """Setter for ``srcHeight`` of element ``i``.
        """
        self._ptr[i].srcHeight = value
    @property
    def srcHeight(self):
        """Getter for ``srcHeight``."""
        return self.get_srcHeight(0)
    @srcHeight.setter
    def srcHeight(self, unsigned int value):
        """Setter for ``srcHeight``."""
        self.set_srcHeight(0,value)
    def get_dstXInBytes(self, i):
        """Getter for ``dstXInBytes`` of element ``i``.
        """
        return self._ptr[i].dstXInBytes
    def set_dstXInBytes(self, i, unsigned int value):
        """Setter for ``dstXInBytes`` of element ``i``.
        """
        self._ptr[i].dstXInBytes = value
    @property
    def dstXInBytes(self):
        """Getter for ``dstXInBytes``."""
        return self.get_dstXInBytes(0)
    @dstXInBytes.setter
    def dstXInBytes(self, unsigned int value):
        """Setter for ``dstXInBytes``."""
        self.set_dstXInBytes(0,value)
    def get_dstY(self, i):
        """Getter for ``dstY`` of element ``i``.
        """
        return self._ptr[i].dstY
    def set_dstY(self, i, unsigned int value):
        """Setter for ``dstY`` of element ``i``.
        """
        self._ptr[i].dstY = value
    @property
    def dstY(self):
        """Getter for ``dstY``."""
        return self.get_dstY(0)
    @dstY.setter
    def dstY(self, unsigned int value):
        """Setter for ``dstY``."""
        self.set_dstY(0,value)
    def get_dstZ(self, i):
        """Getter for ``dstZ`` of element ``i``.
        """
        return self._ptr[i].dstZ
    def set_dstZ(self, i, unsigned int value):
        """Setter for ``dstZ`` of element ``i``.
        """
        self._ptr[i].dstZ = value
    @property
    def dstZ(self):
        """Getter for ``dstZ``."""
        return self.get_dstZ(0)
    @dstZ.setter
    def dstZ(self, unsigned int value):
        """Setter for ``dstZ``."""
        self.set_dstZ(0,value)
    def get_dstLOD(self, i):
        """Getter for ``dstLOD`` of element ``i``.
        """
        return self._ptr[i].dstLOD
    def set_dstLOD(self, i, unsigned int value):
        """Setter for ``dstLOD`` of element ``i``.
        """
        self._ptr[i].dstLOD = value
    @property
    def dstLOD(self):
        """Getter for ``dstLOD``."""
        return self.get_dstLOD(0)
    @dstLOD.setter
    def dstLOD(self, unsigned int value):
        """Setter for ``dstLOD``."""
        self.set_dstLOD(0,value)
    def get_dstMemoryType(self, i):
        """Getter for ``dstMemoryType`` of element ``i``.
        """
        return hipMemoryType(self._ptr[i].dstMemoryType)
    def set_dstMemoryType(self, i, value):
        """Setter for ``dstMemoryType`` of element ``i``.
        """
        if not isinstance(value, hipMemoryType):
            raise TypeError("'value' must be of type 'hipMemoryType'")
        self._ptr[i].dstMemoryType = value.value
    @property
    def dstMemoryType(self):
        """Getter for ``dstMemoryType``."""
        return self.get_dstMemoryType(0)
    @dstMemoryType.setter
    def dstMemoryType(self, value):
        """Setter for ``dstMemoryType``."""
        self.set_dstMemoryType(0,value)
    def get_dstPitch(self, i):
        """Getter for ``dstPitch`` of element ``i``.
        """
        return self._ptr[i].dstPitch
    def set_dstPitch(self, i, unsigned int value):
        """Setter for ``dstPitch`` of element ``i``.
        """
        self._ptr[i].dstPitch = value
    @property
    def dstPitch(self):
        """Getter for ``dstPitch``."""
        return self.get_dstPitch(0)
    @dstPitch.setter
    def dstPitch(self, unsigned int value):
        """Setter for ``dstPitch``."""
        self.set_dstPitch(0,value)
    def get_dstHeight(self, i):
        """Getter for ``dstHeight`` of element ``i``.
        """
        return self._ptr[i].dstHeight
    def set_dstHeight(self, i, unsigned int value):
        """Setter for ``dstHeight`` of element ``i``.
        """
        self._ptr[i].dstHeight = value
    @property
    def dstHeight(self):
        """Getter for ``dstHeight``."""
        return self.get_dstHeight(0)
    @dstHeight.setter
    def dstHeight(self, unsigned int value):
        """Setter for ``dstHeight``."""
        self.set_dstHeight(0,value)
    def get_WidthInBytes(self, i):
        """Getter for ``WidthInBytes`` of element ``i``.
        """
        return self._ptr[i].WidthInBytes
    def set_WidthInBytes(self, i, unsigned int value):
        """Setter for ``WidthInBytes`` of element ``i``.
        """
        self._ptr[i].WidthInBytes = value
    @property
    def WidthInBytes(self):
        """Getter for ``WidthInBytes``."""
        return self.get_WidthInBytes(0)
    @WidthInBytes.setter
    def WidthInBytes(self, unsigned int value):
        """Setter for ``WidthInBytes``."""
        self.set_WidthInBytes(0,value)
    def get_Height(self, i):
        """Getter for ``Height`` of element ``i``.
        """
        return self._ptr[i].Height
    def set_Height(self, i, unsigned int value):
        """Setter for ``Height`` of element ``i``.
        """
        self._ptr[i].Height = value
    @property
    def Height(self):
        """Getter for ``Height``."""
        return self.get_Height(0)
    @Height.setter
    def Height(self, unsigned int value):
        """Setter for ``Height``."""
        self.set_Height(0,value)
    def get_Depth(self, i):
        """Getter for ``Depth`` of element ``i``.
        """
        return self._ptr[i].Depth
    def set_Depth(self, i, unsigned int value):
        """Setter for ``Depth`` of element ``i``.
        """
        self._ptr[i].Depth = value
    @property
    def Depth(self):
        """Getter for ``Depth``."""
        return self.get_Depth(0)
    @Depth.setter
    def Depth(self, unsigned int value):
        """Setter for ``Depth``."""
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
        """Getter for ``normalized`` of element ``i``.
        """
        return self._ptr[i].normalized
    def set_normalized(self, i, int value):
        """Setter for ``normalized`` of element ``i``.
        """
        self._ptr[i].normalized = value
    @property
    def normalized(self):
        """Getter for ``normalized``."""
        return self.get_normalized(0)
    @normalized.setter
    def normalized(self, int value):
        """Setter for ``normalized``."""
        self.set_normalized(0,value)
    def get_readMode(self, i):
        """Getter for ``readMode`` of element ``i``.
        """
        return hipTextureReadMode(self._ptr[i].readMode)
    def set_readMode(self, i, value):
        """Setter for ``readMode`` of element ``i``.
        """
        if not isinstance(value, hipTextureReadMode):
            raise TypeError("'value' must be of type 'hipTextureReadMode'")
        self._ptr[i].readMode = value.value
    @property
    def readMode(self):
        """Getter for ``readMode``."""
        return self.get_readMode(0)
    @readMode.setter
    def readMode(self, value):
        """Setter for ``readMode``."""
        self.set_readMode(0,value)
    def get_filterMode(self, i):
        """Getter for ``filterMode`` of element ``i``.
        """
        return hipTextureFilterMode(self._ptr[i].filterMode)
    def set_filterMode(self, i, value):
        """Setter for ``filterMode`` of element ``i``.
        """
        if not isinstance(value, hipTextureFilterMode):
            raise TypeError("'value' must be of type 'hipTextureFilterMode'")
        self._ptr[i].filterMode = value.value
    @property
    def filterMode(self):
        """Getter for ``filterMode``."""
        return self.get_filterMode(0)
    @filterMode.setter
    def filterMode(self, value):
        """Setter for ``filterMode``."""
        self.set_filterMode(0,value)
    def get_channelDesc(self, i):
        """Getter for ``channelDesc`` of element ``i``.
        """
        return hipChannelFormatDesc.from_ptr(&self._ptr[i].channelDesc)
    @property
    def channelDesc(self):
        """Getter for ``channelDesc``."""
        return self.get_channelDesc(0)
    def get_sRGB(self, i):
        """Getter for ``sRGB`` of element ``i``.
        """
        return self._ptr[i].sRGB
    def set_sRGB(self, i, int value):
        """Setter for ``sRGB`` of element ``i``.
        """
        self._ptr[i].sRGB = value
    @property
    def sRGB(self):
        """Getter for ``sRGB``."""
        return self.get_sRGB(0)
    @sRGB.setter
    def sRGB(self, int value):
        """Setter for ``sRGB``."""
        self.set_sRGB(0,value)
    def get_maxAnisotropy(self, i):
        """Getter for ``maxAnisotropy`` of element ``i``.
        """
        return self._ptr[i].maxAnisotropy
    def set_maxAnisotropy(self, i, unsigned int value):
        """Setter for ``maxAnisotropy`` of element ``i``.
        """
        self._ptr[i].maxAnisotropy = value
    @property
    def maxAnisotropy(self):
        """Getter for ``maxAnisotropy``."""
        return self.get_maxAnisotropy(0)
    @maxAnisotropy.setter
    def maxAnisotropy(self, unsigned int value):
        """Setter for ``maxAnisotropy``."""
        self.set_maxAnisotropy(0,value)
    def get_mipmapFilterMode(self, i):
        """Getter for ``mipmapFilterMode`` of element ``i``.
        """
        return hipTextureFilterMode(self._ptr[i].mipmapFilterMode)
    def set_mipmapFilterMode(self, i, value):
        """Setter for ``mipmapFilterMode`` of element ``i``.
        """
        if not isinstance(value, hipTextureFilterMode):
            raise TypeError("'value' must be of type 'hipTextureFilterMode'")
        self._ptr[i].mipmapFilterMode = value.value
    @property
    def mipmapFilterMode(self):
        """Getter for ``mipmapFilterMode``."""
        return self.get_mipmapFilterMode(0)
    @mipmapFilterMode.setter
    def mipmapFilterMode(self, value):
        """Setter for ``mipmapFilterMode``."""
        self.set_mipmapFilterMode(0,value)
    def get_mipmapLevelBias(self, i):
        """Getter for ``mipmapLevelBias`` of element ``i``.
        """
        return self._ptr[i].mipmapLevelBias
    def set_mipmapLevelBias(self, i, float value):
        """Setter for ``mipmapLevelBias`` of element ``i``.
        """
        self._ptr[i].mipmapLevelBias = value
    @property
    def mipmapLevelBias(self):
        """Getter for ``mipmapLevelBias``."""
        return self.get_mipmapLevelBias(0)
    @mipmapLevelBias.setter
    def mipmapLevelBias(self, float value):
        """Setter for ``mipmapLevelBias``."""
        self.set_mipmapLevelBias(0,value)
    def get_minMipmapLevelClamp(self, i):
        """Getter for ``minMipmapLevelClamp`` of element ``i``.
        """
        return self._ptr[i].minMipmapLevelClamp
    def set_minMipmapLevelClamp(self, i, float value):
        """Setter for ``minMipmapLevelClamp`` of element ``i``.
        """
        self._ptr[i].minMipmapLevelClamp = value
    @property
    def minMipmapLevelClamp(self):
        """Getter for ``minMipmapLevelClamp``."""
        return self.get_minMipmapLevelClamp(0)
    @minMipmapLevelClamp.setter
    def minMipmapLevelClamp(self, float value):
        """Setter for ``minMipmapLevelClamp``."""
        self.set_minMipmapLevelClamp(0,value)
    def get_maxMipmapLevelClamp(self, i):
        """Getter for ``maxMipmapLevelClamp`` of element ``i``.
        """
        return self._ptr[i].maxMipmapLevelClamp
    def set_maxMipmapLevelClamp(self, i, float value):
        """Setter for ``maxMipmapLevelClamp`` of element ``i``.
        """
        self._ptr[i].maxMipmapLevelClamp = value
    @property
    def maxMipmapLevelClamp(self):
        """Getter for ``maxMipmapLevelClamp``."""
        return self.get_maxMipmapLevelClamp(0)
    @maxMipmapLevelClamp.setter
    def maxMipmapLevelClamp(self, float value):
        """Setter for ``maxMipmapLevelClamp``."""
        self.set_maxMipmapLevelClamp(0,value)
    def get_numChannels(self, i):
        """Getter for ``numChannels`` of element ``i``.
        """
        return self._ptr[i].numChannels
    def set_numChannels(self, i, int value):
        """Setter for ``numChannels`` of element ``i``.
        """
        self._ptr[i].numChannels = value
    @property
    def numChannels(self):
        """Getter for ``numChannels``."""
        return self.get_numChannels(0)
    @numChannels.setter
    def numChannels(self, int value):
        """Setter for ``numChannels``."""
        self.set_numChannels(0,value)
    def get_format(self, i):
        """Getter for ``format`` of element ``i``.
        """
        return hipArray_Format(self._ptr[i].format)
    def set_format(self, i, value):
        """Setter for ``format`` of element ``i``.
        """
        if not isinstance(value, hipArray_Format):
            raise TypeError("'value' must be of type 'hipArray_Format'")
        self._ptr[i].format = value.value
    @property
    def format(self):
        """Getter for ``format``."""
        return self.get_format(0)
    @format.setter
    def format(self, value):
        """Setter for ``format``."""
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
    def get_filterMode(self, i):
        """Getter for ``filterMode`` of element ``i``.
        """
        return hipTextureFilterMode(self._ptr[i].filterMode)
    def set_filterMode(self, i, value):
        """Setter for ``filterMode`` of element ``i``.
        """
        if not isinstance(value, hipTextureFilterMode):
            raise TypeError("'value' must be of type 'hipTextureFilterMode'")
        self._ptr[i].filterMode = value.value
    @property
    def filterMode(self):
        """Getter for ``filterMode``."""
        return self.get_filterMode(0)
    @filterMode.setter
    def filterMode(self, value):
        """Setter for ``filterMode``."""
        self.set_filterMode(0,value)
    def get_readMode(self, i):
        """Getter for ``readMode`` of element ``i``.
        """
        return hipTextureReadMode(self._ptr[i].readMode)
    def set_readMode(self, i, value):
        """Setter for ``readMode`` of element ``i``.
        """
        if not isinstance(value, hipTextureReadMode):
            raise TypeError("'value' must be of type 'hipTextureReadMode'")
        self._ptr[i].readMode = value.value
    @property
    def readMode(self):
        """Getter for ``readMode``."""
        return self.get_readMode(0)
    @readMode.setter
    def readMode(self, value):
        """Setter for ``readMode``."""
        self.set_readMode(0,value)
    def get_sRGB(self, i):
        """Getter for ``sRGB`` of element ``i``.
        """
        return self._ptr[i].sRGB
    def set_sRGB(self, i, int value):
        """Setter for ``sRGB`` of element ``i``.
        """
        self._ptr[i].sRGB = value
    @property
    def sRGB(self):
        """Getter for ``sRGB``."""
        return self.get_sRGB(0)
    @sRGB.setter
    def sRGB(self, int value):
        """Setter for ``sRGB``."""
        self.set_sRGB(0,value)
    def get_normalizedCoords(self, i):
        """Getter for ``normalizedCoords`` of element ``i``.
        """
        return self._ptr[i].normalizedCoords
    def set_normalizedCoords(self, i, int value):
        """Setter for ``normalizedCoords`` of element ``i``.
        """
        self._ptr[i].normalizedCoords = value
    @property
    def normalizedCoords(self):
        """Getter for ``normalizedCoords``."""
        return self.get_normalizedCoords(0)
    @normalizedCoords.setter
    def normalizedCoords(self, int value):
        """Setter for ``normalizedCoords``."""
        self.set_normalizedCoords(0,value)
    def get_maxAnisotropy(self, i):
        """Getter for ``maxAnisotropy`` of element ``i``.
        """
        return self._ptr[i].maxAnisotropy
    def set_maxAnisotropy(self, i, unsigned int value):
        """Setter for ``maxAnisotropy`` of element ``i``.
        """
        self._ptr[i].maxAnisotropy = value
    @property
    def maxAnisotropy(self):
        """Getter for ``maxAnisotropy``."""
        return self.get_maxAnisotropy(0)
    @maxAnisotropy.setter
    def maxAnisotropy(self, unsigned int value):
        """Setter for ``maxAnisotropy``."""
        self.set_maxAnisotropy(0,value)
    def get_mipmapFilterMode(self, i):
        """Getter for ``mipmapFilterMode`` of element ``i``.
        """
        return hipTextureFilterMode(self._ptr[i].mipmapFilterMode)
    def set_mipmapFilterMode(self, i, value):
        """Setter for ``mipmapFilterMode`` of element ``i``.
        """
        if not isinstance(value, hipTextureFilterMode):
            raise TypeError("'value' must be of type 'hipTextureFilterMode'")
        self._ptr[i].mipmapFilterMode = value.value
    @property
    def mipmapFilterMode(self):
        """Getter for ``mipmapFilterMode``."""
        return self.get_mipmapFilterMode(0)
    @mipmapFilterMode.setter
    def mipmapFilterMode(self, value):
        """Setter for ``mipmapFilterMode``."""
        self.set_mipmapFilterMode(0,value)
    def get_mipmapLevelBias(self, i):
        """Getter for ``mipmapLevelBias`` of element ``i``.
        """
        return self._ptr[i].mipmapLevelBias
    def set_mipmapLevelBias(self, i, float value):
        """Setter for ``mipmapLevelBias`` of element ``i``.
        """
        self._ptr[i].mipmapLevelBias = value
    @property
    def mipmapLevelBias(self):
        """Getter for ``mipmapLevelBias``."""
        return self.get_mipmapLevelBias(0)
    @mipmapLevelBias.setter
    def mipmapLevelBias(self, float value):
        """Setter for ``mipmapLevelBias``."""
        self.set_mipmapLevelBias(0,value)
    def get_minMipmapLevelClamp(self, i):
        """Getter for ``minMipmapLevelClamp`` of element ``i``.
        """
        return self._ptr[i].minMipmapLevelClamp
    def set_minMipmapLevelClamp(self, i, float value):
        """Setter for ``minMipmapLevelClamp`` of element ``i``.
        """
        self._ptr[i].minMipmapLevelClamp = value
    @property
    def minMipmapLevelClamp(self):
        """Getter for ``minMipmapLevelClamp``."""
        return self.get_minMipmapLevelClamp(0)
    @minMipmapLevelClamp.setter
    def minMipmapLevelClamp(self, float value):
        """Setter for ``minMipmapLevelClamp``."""
        self.set_minMipmapLevelClamp(0,value)
    def get_maxMipmapLevelClamp(self, i):
        """Getter for ``maxMipmapLevelClamp`` of element ``i``.
        """
        return self._ptr[i].maxMipmapLevelClamp
    def set_maxMipmapLevelClamp(self, i, float value):
        """Setter for ``maxMipmapLevelClamp`` of element ``i``.
        """
        self._ptr[i].maxMipmapLevelClamp = value
    @property
    def maxMipmapLevelClamp(self):
        """Getter for ``maxMipmapLevelClamp``."""
        return self.get_maxMipmapLevelClamp(0)
    @maxMipmapLevelClamp.setter
    def maxMipmapLevelClamp(self, float value):
        """Setter for ``maxMipmapLevelClamp``."""
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
    def get_binaryVersion(self, i):
        """Getter for ``binaryVersion`` of element ``i``.
        """
        return self._ptr[i].binaryVersion
    def set_binaryVersion(self, i, int value):
        """Setter for ``binaryVersion`` of element ``i``.
        """
        self._ptr[i].binaryVersion = value
    @property
    def binaryVersion(self):
        """Getter for ``binaryVersion``."""
        return self.get_binaryVersion(0)
    @binaryVersion.setter
    def binaryVersion(self, int value):
        """Setter for ``binaryVersion``."""
        self.set_binaryVersion(0,value)
    def get_cacheModeCA(self, i):
        """Getter for ``cacheModeCA`` of element ``i``.
        """
        return self._ptr[i].cacheModeCA
    def set_cacheModeCA(self, i, int value):
        """Setter for ``cacheModeCA`` of element ``i``.
        """
        self._ptr[i].cacheModeCA = value
    @property
    def cacheModeCA(self):
        """Getter for ``cacheModeCA``."""
        return self.get_cacheModeCA(0)
    @cacheModeCA.setter
    def cacheModeCA(self, int value):
        """Setter for ``cacheModeCA``."""
        self.set_cacheModeCA(0,value)
    def get_constSizeBytes(self, i):
        """Getter for ``constSizeBytes`` of element ``i``.
        """
        return self._ptr[i].constSizeBytes
    def set_constSizeBytes(self, i, int value):
        """Setter for ``constSizeBytes`` of element ``i``.
        """
        self._ptr[i].constSizeBytes = value
    @property
    def constSizeBytes(self):
        """Getter for ``constSizeBytes``."""
        return self.get_constSizeBytes(0)
    @constSizeBytes.setter
    def constSizeBytes(self, int value):
        """Setter for ``constSizeBytes``."""
        self.set_constSizeBytes(0,value)
    def get_localSizeBytes(self, i):
        """Getter for ``localSizeBytes`` of element ``i``.
        """
        return self._ptr[i].localSizeBytes
    def set_localSizeBytes(self, i, int value):
        """Setter for ``localSizeBytes`` of element ``i``.
        """
        self._ptr[i].localSizeBytes = value
    @property
    def localSizeBytes(self):
        """Getter for ``localSizeBytes``."""
        return self.get_localSizeBytes(0)
    @localSizeBytes.setter
    def localSizeBytes(self, int value):
        """Setter for ``localSizeBytes``."""
        self.set_localSizeBytes(0,value)
    def get_maxDynamicSharedSizeBytes(self, i):
        """Getter for ``maxDynamicSharedSizeBytes`` of element ``i``.
        """
        return self._ptr[i].maxDynamicSharedSizeBytes
    def set_maxDynamicSharedSizeBytes(self, i, int value):
        """Setter for ``maxDynamicSharedSizeBytes`` of element ``i``.
        """
        self._ptr[i].maxDynamicSharedSizeBytes = value
    @property
    def maxDynamicSharedSizeBytes(self):
        """Getter for ``maxDynamicSharedSizeBytes``."""
        return self.get_maxDynamicSharedSizeBytes(0)
    @maxDynamicSharedSizeBytes.setter
    def maxDynamicSharedSizeBytes(self, int value):
        """Setter for ``maxDynamicSharedSizeBytes``."""
        self.set_maxDynamicSharedSizeBytes(0,value)
    def get_maxThreadsPerBlock(self, i):
        """Getter for ``maxThreadsPerBlock`` of element ``i``.
        """
        return self._ptr[i].maxThreadsPerBlock
    def set_maxThreadsPerBlock(self, i, int value):
        """Setter for ``maxThreadsPerBlock`` of element ``i``.
        """
        self._ptr[i].maxThreadsPerBlock = value
    @property
    def maxThreadsPerBlock(self):
        """Getter for ``maxThreadsPerBlock``."""
        return self.get_maxThreadsPerBlock(0)
    @maxThreadsPerBlock.setter
    def maxThreadsPerBlock(self, int value):
        """Setter for ``maxThreadsPerBlock``."""
        self.set_maxThreadsPerBlock(0,value)
    def get_numRegs(self, i):
        """Getter for ``numRegs`` of element ``i``.
        """
        return self._ptr[i].numRegs
    def set_numRegs(self, i, int value):
        """Setter for ``numRegs`` of element ``i``.
        """
        self._ptr[i].numRegs = value
    @property
    def numRegs(self):
        """Getter for ``numRegs``."""
        return self.get_numRegs(0)
    @numRegs.setter
    def numRegs(self, int value):
        """Setter for ``numRegs``."""
        self.set_numRegs(0,value)
    def get_preferredShmemCarveout(self, i):
        """Getter for ``preferredShmemCarveout`` of element ``i``.
        """
        return self._ptr[i].preferredShmemCarveout
    def set_preferredShmemCarveout(self, i, int value):
        """Setter for ``preferredShmemCarveout`` of element ``i``.
        """
        self._ptr[i].preferredShmemCarveout = value
    @property
    def preferredShmemCarveout(self):
        """Getter for ``preferredShmemCarveout``."""
        return self.get_preferredShmemCarveout(0)
    @preferredShmemCarveout.setter
    def preferredShmemCarveout(self, int value):
        """Setter for ``preferredShmemCarveout``."""
        self.set_preferredShmemCarveout(0,value)
    def get_ptxVersion(self, i):
        """Getter for ``ptxVersion`` of element ``i``.
        """
        return self._ptr[i].ptxVersion
    def set_ptxVersion(self, i, int value):
        """Setter for ``ptxVersion`` of element ``i``.
        """
        self._ptr[i].ptxVersion = value
    @property
    def ptxVersion(self):
        """Getter for ``ptxVersion``."""
        return self.get_ptxVersion(0)
    @ptxVersion.setter
    def ptxVersion(self, int value):
        """Setter for ``ptxVersion``."""
        self.set_ptxVersion(0,value)
    def get_sharedSizeBytes(self, i):
        """Getter for ``sharedSizeBytes`` of element ``i``.
        """
        return self._ptr[i].sharedSizeBytes
    def set_sharedSizeBytes(self, i, int value):
        """Setter for ``sharedSizeBytes`` of element ``i``.
        """
        self._ptr[i].sharedSizeBytes = value
    @property
    def sharedSizeBytes(self):
        """Getter for ``sharedSizeBytes``."""
        return self.get_sharedSizeBytes(0)
    @sharedSizeBytes.setter
    def sharedSizeBytes(self, int value):
        """Setter for ``sharedSizeBytes``."""
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
        """Getter for ``type`` of element ``i``.
        """
        return hipMemLocationType(self._ptr[i].type)
    def set_type(self, i, value):
        """Setter for ``type`` of element ``i``.
        """
        if not isinstance(value, hipMemLocationType):
            raise TypeError("'value' must be of type 'hipMemLocationType'")
        self._ptr[i].type = value.value
    @property
    def type(self):
        """Getter for ``type``."""
        return self.get_type(0)
    @type.setter
    def type(self, value):
        """Setter for ``type``."""
        self.set_type(0,value)
    def get_id(self, i):
        """Getter for ``id`` of element ``i``.
        """
        return self._ptr[i].id
    def set_id(self, i, int value):
        """Setter for ``id`` of element ``i``.
        """
        self._ptr[i].id = value
    @property
    def id(self):
        """Getter for ``id``."""
        return self.get_id(0)
    @id.setter
    def id(self, int value):
        """Setter for ``id``."""
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
        """Getter for ``location`` of element ``i``.
        """
        return hipMemLocation.from_ptr(&self._ptr[i].location)
    @property
    def location(self):
        """Getter for ``location``."""
        return self.get_location(0)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return hipMemAccessFlags(self._ptr[i].flags)
    def set_flags(self, i, value):
        """Setter for ``flags`` of element ``i``.
        """
        if not isinstance(value, hipMemAccessFlags):
            raise TypeError("'value' must be of type 'hipMemAccessFlags'")
        self._ptr[i].flags = value.value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, value):
        """Setter for ``flags``."""
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
        """Getter for ``allocType`` of element ``i``.
        """
        return hipMemAllocationType(self._ptr[i].allocType)
    def set_allocType(self, i, value):
        """Setter for ``allocType`` of element ``i``.
        """
        if not isinstance(value, hipMemAllocationType):
            raise TypeError("'value' must be of type 'hipMemAllocationType'")
        self._ptr[i].allocType = value.value
    @property
    def allocType(self):
        """Getter for ``allocType``."""
        return self.get_allocType(0)
    @allocType.setter
    def allocType(self, value):
        """Setter for ``allocType``."""
        self.set_allocType(0,value)
    def get_handleTypes(self, i):
        """Getter for ``handleTypes`` of element ``i``.
        """
        return hipMemAllocationHandleType(self._ptr[i].handleTypes)
    def set_handleTypes(self, i, value):
        """Setter for ``handleTypes`` of element ``i``.
        """
        if not isinstance(value, hipMemAllocationHandleType):
            raise TypeError("'value' must be of type 'hipMemAllocationHandleType'")
        self._ptr[i].handleTypes = value.value
    @property
    def handleTypes(self):
        """Getter for ``handleTypes``."""
        return self.get_handleTypes(0)
    @handleTypes.setter
    def handleTypes(self, value):
        """Setter for ``handleTypes``."""
        self.set_handleTypes(0,value)
    def get_location(self, i):
        """Getter for ``location`` of element ``i``.
        """
        return hipMemLocation.from_ptr(&self._ptr[i].location)
    @property
    def location(self):
        """Getter for ``location``."""
        return self.get_location(0)



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
    def get_x(self, i):
        """Getter for ``x`` of element ``i``.
        """
        return self._ptr[i].x
    def set_x(self, i, uint32_t value):
        """Setter for ``x`` of element ``i``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        """Getter for ``x``."""
        return self.get_x(0)
    @x.setter
    def x(self, uint32_t value):
        """Setter for ``x``."""
        self.set_x(0,value)
    def get_y(self, i):
        """Getter for ``y`` of element ``i``.
        """
        return self._ptr[i].y
    def set_y(self, i, uint32_t value):
        """Setter for ``y`` of element ``i``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        """Getter for ``y``."""
        return self.get_y(0)
    @y.setter
    def y(self, uint32_t value):
        """Setter for ``y``."""
        self.set_y(0,value)
    def get_z(self, i):
        """Getter for ``z`` of element ``i``.
        """
        return self._ptr[i].z
    def set_z(self, i, uint32_t value):
        """Setter for ``z`` of element ``i``.
        """
        self._ptr[i].z = value
    @property
    def z(self):
        """Getter for ``z``."""
        return self.get_z(0)
    @z.setter
    def z(self, uint32_t value):
        """Setter for ``z``."""
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
        """Getter for ``gridDim`` of element ``i``.
        """
        return dim3.from_ptr(&self._ptr[i].gridDim)
    @property
    def gridDim(self):
        """Getter for ``gridDim``."""
        return self.get_gridDim(0)
    def get_blockDim(self, i):
        """Getter for ``blockDim`` of element ``i``.
        """
        return dim3.from_ptr(&self._ptr[i].blockDim)
    @property
    def blockDim(self):
        """Getter for ``blockDim``."""
        return self.get_blockDim(0)
    def get_sharedMem(self, i):
        """Getter for ``sharedMem`` of element ``i``.
        """
        return self._ptr[i].sharedMem
    def set_sharedMem(self, i, int value):
        """Setter for ``sharedMem`` of element ``i``.
        """
        self._ptr[i].sharedMem = value
    @property
    def sharedMem(self):
        """Getter for ``sharedMem``."""
        return self.get_sharedMem(0)
    @sharedMem.setter
    def sharedMem(self, int value):
        """Setter for ``sharedMem``."""
        self.set_sharedMem(0,value)


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
    def get_fd(self, i):
        """Getter for ``fd`` of element ``i``.
        """
        return self._ptr[i].fd
    def set_fd(self, i, int value):
        """Setter for ``fd`` of element ``i``.
        """
        self._ptr[i].fd = value
    @property
    def fd(self):
        """Getter for ``fd``."""
        return self.get_fd(0)
    @fd.setter
    def fd(self, int value):
        """Setter for ``fd``."""
        self.set_fd(0,value)
    def get_win32(self, i):
        """Getter for ``win32`` of element ``i``.
        """
        return hipExternalMemoryHandleDesc_st_union_0_struct_0.from_ptr(&self._ptr[i].win32)
    @property
    def win32(self):
        """Getter for ``win32``."""
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
        """Getter for ``type`` of element ``i``.
        """
        return hipExternalMemoryHandleType_enum(self._ptr[i].type)
    def set_type(self, i, value):
        """Setter for ``type`` of element ``i``.
        """
        if not isinstance(value, hipExternalMemoryHandleType_enum):
            raise TypeError("'value' must be of type 'hipExternalMemoryHandleType_enum'")
        self._ptr[i].type = value.value
    @property
    def type(self):
        """Getter for ``type``."""
        return self.get_type(0)
    @type.setter
    def type(self, value):
        """Setter for ``type``."""
        self.set_type(0,value)
    def get_handle(self, i):
        """Getter for ``handle`` of element ``i``.
        """
        return hipExternalMemoryHandleDesc_st_union_0.from_ptr(&self._ptr[i].handle)
    @property
    def handle(self):
        """Getter for ``handle``."""
        return self.get_handle(0)
    def get_size(self, i):
        """Getter for ``size`` of element ``i``.
        """
        return self._ptr[i].size
    def set_size(self, i, unsigned long long value):
        """Setter for ``size`` of element ``i``.
        """
        self._ptr[i].size = value
    @property
    def size(self):
        """Getter for ``size``."""
        return self.get_size(0)
    @size.setter
    def size(self, unsigned long long value):
        """Setter for ``size``."""
        self.set_size(0,value)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Setter for ``flags`` of element ``i``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        """Setter for ``flags``."""
        self.set_flags(0,value)



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
        """Getter for ``offset`` of element ``i``.
        """
        return self._ptr[i].offset
    def set_offset(self, i, unsigned long long value):
        """Setter for ``offset`` of element ``i``.
        """
        self._ptr[i].offset = value
    @property
    def offset(self):
        """Getter for ``offset``."""
        return self.get_offset(0)
    @offset.setter
    def offset(self, unsigned long long value):
        """Setter for ``offset``."""
        self.set_offset(0,value)
    def get_size(self, i):
        """Getter for ``size`` of element ``i``.
        """
        return self._ptr[i].size
    def set_size(self, i, unsigned long long value):
        """Setter for ``size`` of element ``i``.
        """
        self._ptr[i].size = value
    @property
    def size(self):
        """Getter for ``size``."""
        return self.get_size(0)
    @size.setter
    def size(self, unsigned long long value):
        """Setter for ``size``."""
        self.set_size(0,value)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Setter for ``flags`` of element ``i``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        """Setter for ``flags``."""
        self.set_flags(0,value)


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
    def get_fd(self, i):
        """Getter for ``fd`` of element ``i``.
        """
        return self._ptr[i].fd
    def set_fd(self, i, int value):
        """Setter for ``fd`` of element ``i``.
        """
        self._ptr[i].fd = value
    @property
    def fd(self):
        """Getter for ``fd``."""
        return self.get_fd(0)
    @fd.setter
    def fd(self, int value):
        """Setter for ``fd``."""
        self.set_fd(0,value)
    def get_win32(self, i):
        """Getter for ``win32`` of element ``i``.
        """
        return hipExternalSemaphoreHandleDesc_st_union_0_struct_0.from_ptr(&self._ptr[i].win32)
    @property
    def win32(self):
        """Getter for ``win32``."""
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
        """Getter for ``type`` of element ``i``.
        """
        return hipExternalSemaphoreHandleType_enum(self._ptr[i].type)
    def set_type(self, i, value):
        """Setter for ``type`` of element ``i``.
        """
        if not isinstance(value, hipExternalSemaphoreHandleType_enum):
            raise TypeError("'value' must be of type 'hipExternalSemaphoreHandleType_enum'")
        self._ptr[i].type = value.value
    @property
    def type(self):
        """Getter for ``type``."""
        return self.get_type(0)
    @type.setter
    def type(self, value):
        """Setter for ``type``."""
        self.set_type(0,value)
    def get_handle(self, i):
        """Getter for ``handle`` of element ``i``.
        """
        return hipExternalSemaphoreHandleDesc_st_union_0.from_ptr(&self._ptr[i].handle)
    @property
    def handle(self):
        """Getter for ``handle``."""
        return self.get_handle(0)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Setter for ``flags`` of element ``i``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        """Setter for ``flags``."""
        self.set_flags(0,value)



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
        """Getter for ``value`` of element ``i``.
        """
        return self._ptr[i].value
    def set_value(self, i, unsigned long long value):
        """Setter for ``value`` of element ``i``.
        """
        self._ptr[i].value = value
    @property
    def value(self):
        """Getter for ``value``."""
        return self.get_value(0)
    @value.setter
    def value(self, unsigned long long value):
        """Setter for ``value``."""
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
        """Getter for ``key`` of element ``i``.
        """
        return self._ptr[i].key
    def set_key(self, i, unsigned long long value):
        """Setter for ``key`` of element ``i``.
        """
        self._ptr[i].key = value
    @property
    def key(self):
        """Getter for ``key``."""
        return self.get_key(0)
    @key.setter
    def key(self, unsigned long long value):
        """Setter for ``key``."""
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
        """Getter for ``fence`` of element ``i``.
        """
        return hipExternalSemaphoreSignalParams_st_struct_0_struct_0.from_ptr(&self._ptr[i].fence)
    @property
    def fence(self):
        """Getter for ``fence``."""
        return self.get_fence(0)
    def get_keyedMutex(self, i):
        """Getter for ``keyedMutex`` of element ``i``.
        """
        return hipExternalSemaphoreSignalParams_st_struct_0_struct_1.from_ptr(&self._ptr[i].keyedMutex)
    @property
    def keyedMutex(self):
        """Getter for ``keyedMutex``."""
        return self.get_keyedMutex(0)



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
        """Getter for ``params`` of element ``i``.
        """
        return hipExternalSemaphoreSignalParams_st_struct_0.from_ptr(&self._ptr[i].params)
    @property
    def params(self):
        """Getter for ``params``."""
        return self.get_params(0)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Setter for ``flags`` of element ``i``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        """Setter for ``flags``."""
        self.set_flags(0,value)



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
        """Getter for ``value`` of element ``i``.
        """
        return self._ptr[i].value
    def set_value(self, i, unsigned long long value):
        """Setter for ``value`` of element ``i``.
        """
        self._ptr[i].value = value
    @property
    def value(self):
        """Getter for ``value``."""
        return self.get_value(0)
    @value.setter
    def value(self, unsigned long long value):
        """Setter for ``value``."""
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
        """Getter for ``key`` of element ``i``.
        """
        return self._ptr[i].key
    def set_key(self, i, unsigned long long value):
        """Setter for ``key`` of element ``i``.
        """
        self._ptr[i].key = value
    @property
    def key(self):
        """Getter for ``key``."""
        return self.get_key(0)
    @key.setter
    def key(self, unsigned long long value):
        """Setter for ``key``."""
        self.set_key(0,value)
    def get_timeoutMs(self, i):
        """Getter for ``timeoutMs`` of element ``i``.
        """
        return self._ptr[i].timeoutMs
    def set_timeoutMs(self, i, unsigned int value):
        """Setter for ``timeoutMs`` of element ``i``.
        """
        self._ptr[i].timeoutMs = value
    @property
    def timeoutMs(self):
        """Getter for ``timeoutMs``."""
        return self.get_timeoutMs(0)
    @timeoutMs.setter
    def timeoutMs(self, unsigned int value):
        """Setter for ``timeoutMs``."""
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
        """Getter for ``fence`` of element ``i``.
        """
        return hipExternalSemaphoreWaitParams_st_struct_0_struct_0.from_ptr(&self._ptr[i].fence)
    @property
    def fence(self):
        """Getter for ``fence``."""
        return self.get_fence(0)
    def get_keyedMutex(self, i):
        """Getter for ``keyedMutex`` of element ``i``.
        """
        return hipExternalSemaphoreWaitParams_st_struct_0_struct_1.from_ptr(&self._ptr[i].keyedMutex)
    @property
    def keyedMutex(self):
        """Getter for ``keyedMutex``."""
        return self.get_keyedMutex(0)



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
        """Getter for ``params`` of element ``i``.
        """
        return hipExternalSemaphoreWaitParams_st_struct_0.from_ptr(&self._ptr[i].params)
    @property
    def params(self):
        """Getter for ``params``."""
        return self.get_params(0)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Setter for ``flags`` of element ``i``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        """Setter for ``flags``."""
        self.set_flags(0,value)


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
    def get_blockDim(self, i):
        """Getter for ``blockDim`` of element ``i``.
        """
        return dim3.from_ptr(&self._ptr[i].blockDim)
    @property
    def blockDim(self):
        """Getter for ``blockDim``."""
        return self.get_blockDim(0)
    def get_gridDim(self, i):
        """Getter for ``gridDim`` of element ``i``.
        """
        return dim3.from_ptr(&self._ptr[i].gridDim)
    @property
    def gridDim(self):
        """Getter for ``gridDim``."""
        return self.get_gridDim(0)
    def get_sharedMemBytes(self, i):
        """Getter for ``sharedMemBytes`` of element ``i``.
        """
        return self._ptr[i].sharedMemBytes
    def set_sharedMemBytes(self, i, unsigned int value):
        """Setter for ``sharedMemBytes`` of element ``i``.
        """
        self._ptr[i].sharedMemBytes = value
    @property
    def sharedMemBytes(self):
        """Getter for ``sharedMemBytes``."""
        return self.get_sharedMemBytes(0)
    @sharedMemBytes.setter
    def sharedMemBytes(self, unsigned int value):
        """Setter for ``sharedMemBytes``."""
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
        """Getter for ``elementSize`` of element ``i``.
        """
        return self._ptr[i].elementSize
    def set_elementSize(self, i, unsigned int value):
        """Setter for ``elementSize`` of element ``i``.
        """
        self._ptr[i].elementSize = value
    @property
    def elementSize(self):
        """Getter for ``elementSize``."""
        return self.get_elementSize(0)
    @elementSize.setter
    def elementSize(self, unsigned int value):
        """Setter for ``elementSize``."""
        self.set_elementSize(0,value)
    def get_height(self, i):
        """Getter for ``height`` of element ``i``.
        """
        return self._ptr[i].height
    def set_height(self, i, int value):
        """Setter for ``height`` of element ``i``.
        """
        self._ptr[i].height = value
    @property
    def height(self):
        """Getter for ``height``."""
        return self.get_height(0)
    @height.setter
    def height(self, int value):
        """Setter for ``height``."""
        self.set_height(0,value)
    def get_pitch(self, i):
        """Getter for ``pitch`` of element ``i``.
        """
        return self._ptr[i].pitch
    def set_pitch(self, i, int value):
        """Setter for ``pitch`` of element ``i``.
        """
        self._ptr[i].pitch = value
    @property
    def pitch(self):
        """Getter for ``pitch``."""
        return self.get_pitch(0)
    @pitch.setter
    def pitch(self, int value):
        """Setter for ``pitch``."""
        self.set_pitch(0,value)
    def get_value(self, i):
        """Getter for ``value`` of element ``i``.
        """
        return self._ptr[i].value
    def set_value(self, i, unsigned int value):
        """Setter for ``value`` of element ``i``.
        """
        self._ptr[i].value = value
    @property
    def value(self):
        """Getter for ``value``."""
        return self.get_value(0)
    @value.setter
    def value(self, unsigned int value):
        """Setter for ``value``."""
        self.set_value(0,value)
    def get_width(self, i):
        """Getter for ``width`` of element ``i``.
        """
        return self._ptr[i].width
    def set_width(self, i, int value):
        """Setter for ``width`` of element ``i``.
        """
        self._ptr[i].width = value
    @property
    def width(self):
        """Getter for ``width``."""
        return self.get_width(0)
    @width.setter
    def width(self, int value):
        """Setter for ``width``."""
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
        """Getter for ``hitProp`` of element ``i``.
        """
        return hipAccessProperty(self._ptr[i].hitProp)
    def set_hitProp(self, i, value):
        """Setter for ``hitProp`` of element ``i``.
        """
        if not isinstance(value, hipAccessProperty):
            raise TypeError("'value' must be of type 'hipAccessProperty'")
        self._ptr[i].hitProp = value.value
    @property
    def hitProp(self):
        """Getter for ``hitProp``."""
        return self.get_hitProp(0)
    @hitProp.setter
    def hitProp(self, value):
        """Setter for ``hitProp``."""
        self.set_hitProp(0,value)
    def get_hitRatio(self, i):
        """Getter for ``hitRatio`` of element ``i``.
        """
        return self._ptr[i].hitRatio
    def set_hitRatio(self, i, float value):
        """Setter for ``hitRatio`` of element ``i``.
        """
        self._ptr[i].hitRatio = value
    @property
    def hitRatio(self):
        """Getter for ``hitRatio``."""
        return self.get_hitRatio(0)
    @hitRatio.setter
    def hitRatio(self, float value):
        """Setter for ``hitRatio``."""
        self.set_hitRatio(0,value)
    def get_missProp(self, i):
        """Getter for ``missProp`` of element ``i``.
        """
        return hipAccessProperty(self._ptr[i].missProp)
    def set_missProp(self, i, value):
        """Setter for ``missProp`` of element ``i``.
        """
        if not isinstance(value, hipAccessProperty):
            raise TypeError("'value' must be of type 'hipAccessProperty'")
        self._ptr[i].missProp = value.value
    @property
    def missProp(self):
        """Getter for ``missProp``."""
        return self.get_missProp(0)
    @missProp.setter
    def missProp(self, value):
        """Setter for ``missProp``."""
        self.set_missProp(0,value)
    def get_num_bytes(self, i):
        """Getter for ``num_bytes`` of element ``i``.
        """
        return self._ptr[i].num_bytes
    def set_num_bytes(self, i, int value):
        """Setter for ``num_bytes`` of element ``i``.
        """
        self._ptr[i].num_bytes = value
    @property
    def num_bytes(self):
        """Getter for ``num_bytes``."""
        return self.get_num_bytes(0)
    @num_bytes.setter
    def num_bytes(self, int value):
        """Setter for ``num_bytes``."""
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
        """Getter for ``accessPolicyWindow`` of element ``i``.
        """
        return hipAccessPolicyWindow.from_ptr(&self._ptr[i].accessPolicyWindow)
    @property
    def accessPolicyWindow(self):
        """Getter for ``accessPolicyWindow``."""
        return self.get_accessPolicyWindow(0)
    def get_cooperative(self, i):
        """Getter for ``cooperative`` of element ``i``.
        """
        return self._ptr[i].cooperative
    def set_cooperative(self, i, int value):
        """Setter for ``cooperative`` of element ``i``.
        """
        self._ptr[i].cooperative = value
    @property
    def cooperative(self):
        """Getter for ``cooperative``."""
        return self.get_cooperative(0)
    @cooperative.setter
    def cooperative(self, int value):
        """Setter for ``cooperative``."""
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
        """Getter for ``compressionType`` of element ``i``.
        """
        return self._ptr[i].compressionType
    def set_compressionType(self, i, unsigned char value):
        """Setter for ``compressionType`` of element ``i``.
        """
        self._ptr[i].compressionType = value
    @property
    def compressionType(self):
        """Getter for ``compressionType``."""
        return self.get_compressionType(0)
    @compressionType.setter
    def compressionType(self, unsigned char value):
        """Setter for ``compressionType``."""
        self.set_compressionType(0,value)
    def get_gpuDirectRDMACapable(self, i):
        """Getter for ``gpuDirectRDMACapable`` of element ``i``.
        """
        return self._ptr[i].gpuDirectRDMACapable
    def set_gpuDirectRDMACapable(self, i, unsigned char value):
        """Setter for ``gpuDirectRDMACapable`` of element ``i``.
        """
        self._ptr[i].gpuDirectRDMACapable = value
    @property
    def gpuDirectRDMACapable(self):
        """Getter for ``gpuDirectRDMACapable``."""
        return self.get_gpuDirectRDMACapable(0)
    @gpuDirectRDMACapable.setter
    def gpuDirectRDMACapable(self, unsigned char value):
        """Setter for ``gpuDirectRDMACapable``."""
        self.set_gpuDirectRDMACapable(0,value)
    def get_usage(self, i):
        """Getter for ``usage`` of element ``i``.
        """
        return self._ptr[i].usage
    def set_usage(self, i, unsigned short value):
        """Setter for ``usage`` of element ``i``.
        """
        self._ptr[i].usage = value
    @property
    def usage(self):
        """Getter for ``usage``."""
        return self.get_usage(0)
    @usage.setter
    def usage(self, unsigned short value):
        """Setter for ``usage``."""
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
        """Getter for ``type`` of element ``i``.
        """
        return hipMemAllocationType(self._ptr[i].type)
    def set_type(self, i, value):
        """Setter for ``type`` of element ``i``.
        """
        if not isinstance(value, hipMemAllocationType):
            raise TypeError("'value' must be of type 'hipMemAllocationType'")
        self._ptr[i].type = value.value
    @property
    def type(self):
        """Getter for ``type``."""
        return self.get_type(0)
    @type.setter
    def type(self, value):
        """Setter for ``type``."""
        self.set_type(0,value)
    def get_requestedHandleType(self, i):
        """Getter for ``requestedHandleType`` of element ``i``.
        """
        return hipMemAllocationHandleType(self._ptr[i].requestedHandleType)
    def set_requestedHandleType(self, i, value):
        """Setter for ``requestedHandleType`` of element ``i``.
        """
        if not isinstance(value, hipMemAllocationHandleType):
            raise TypeError("'value' must be of type 'hipMemAllocationHandleType'")
        self._ptr[i].requestedHandleType = value.value
    @property
    def requestedHandleType(self):
        """Getter for ``requestedHandleType``."""
        return self.get_requestedHandleType(0)
    @requestedHandleType.setter
    def requestedHandleType(self, value):
        """Setter for ``requestedHandleType``."""
        self.set_requestedHandleType(0,value)
    def get_location(self, i):
        """Getter for ``location`` of element ``i``.
        """
        return hipMemLocation.from_ptr(&self._ptr[i].location)
    @property
    def location(self):
        """Getter for ``location``."""
        return self.get_location(0)
    def get_allocFlags(self, i):
        """Getter for ``allocFlags`` of element ``i``.
        """
        return hipMemAllocationProp_struct_0.from_ptr(&self._ptr[i].allocFlags)
    @property
    def allocFlags(self):
        """Getter for ``allocFlags``."""
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
        """Getter for ``mipmap`` of element ``i``.
        """
        return hipMipmappedArray.from_ptr(&self._ptr[i].mipmap)
    @property
    def mipmap(self):
        """Getter for ``mipmap``."""
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
        """Getter for ``level`` of element ``i``.
        """
        return self._ptr[i].level
    def set_level(self, i, unsigned int value):
        """Setter for ``level`` of element ``i``.
        """
        self._ptr[i].level = value
    @property
    def level(self):
        """Getter for ``level``."""
        return self.get_level(0)
    @level.setter
    def level(self, unsigned int value):
        """Setter for ``level``."""
        self.set_level(0,value)
    def get_layer(self, i):
        """Getter for ``layer`` of element ``i``.
        """
        return self._ptr[i].layer
    def set_layer(self, i, unsigned int value):
        """Setter for ``layer`` of element ``i``.
        """
        self._ptr[i].layer = value
    @property
    def layer(self):
        """Getter for ``layer``."""
        return self.get_layer(0)
    @layer.setter
    def layer(self, unsigned int value):
        """Setter for ``layer``."""
        self.set_layer(0,value)
    def get_offsetX(self, i):
        """Getter for ``offsetX`` of element ``i``.
        """
        return self._ptr[i].offsetX
    def set_offsetX(self, i, unsigned int value):
        """Setter for ``offsetX`` of element ``i``.
        """
        self._ptr[i].offsetX = value
    @property
    def offsetX(self):
        """Getter for ``offsetX``."""
        return self.get_offsetX(0)
    @offsetX.setter
    def offsetX(self, unsigned int value):
        """Setter for ``offsetX``."""
        self.set_offsetX(0,value)
    def get_offsetY(self, i):
        """Getter for ``offsetY`` of element ``i``.
        """
        return self._ptr[i].offsetY
    def set_offsetY(self, i, unsigned int value):
        """Setter for ``offsetY`` of element ``i``.
        """
        self._ptr[i].offsetY = value
    @property
    def offsetY(self):
        """Getter for ``offsetY``."""
        return self.get_offsetY(0)
    @offsetY.setter
    def offsetY(self, unsigned int value):
        """Setter for ``offsetY``."""
        self.set_offsetY(0,value)
    def get_offsetZ(self, i):
        """Getter for ``offsetZ`` of element ``i``.
        """
        return self._ptr[i].offsetZ
    def set_offsetZ(self, i, unsigned int value):
        """Setter for ``offsetZ`` of element ``i``.
        """
        self._ptr[i].offsetZ = value
    @property
    def offsetZ(self):
        """Getter for ``offsetZ``."""
        return self.get_offsetZ(0)
    @offsetZ.setter
    def offsetZ(self, unsigned int value):
        """Setter for ``offsetZ``."""
        self.set_offsetZ(0,value)
    def get_extentWidth(self, i):
        """Getter for ``extentWidth`` of element ``i``.
        """
        return self._ptr[i].extentWidth
    def set_extentWidth(self, i, unsigned int value):
        """Setter for ``extentWidth`` of element ``i``.
        """
        self._ptr[i].extentWidth = value
    @property
    def extentWidth(self):
        """Getter for ``extentWidth``."""
        return self.get_extentWidth(0)
    @extentWidth.setter
    def extentWidth(self, unsigned int value):
        """Setter for ``extentWidth``."""
        self.set_extentWidth(0,value)
    def get_extentHeight(self, i):
        """Getter for ``extentHeight`` of element ``i``.
        """
        return self._ptr[i].extentHeight
    def set_extentHeight(self, i, unsigned int value):
        """Setter for ``extentHeight`` of element ``i``.
        """
        self._ptr[i].extentHeight = value
    @property
    def extentHeight(self):
        """Getter for ``extentHeight``."""
        return self.get_extentHeight(0)
    @extentHeight.setter
    def extentHeight(self, unsigned int value):
        """Setter for ``extentHeight``."""
        self.set_extentHeight(0,value)
    def get_extentDepth(self, i):
        """Getter for ``extentDepth`` of element ``i``.
        """
        return self._ptr[i].extentDepth
    def set_extentDepth(self, i, unsigned int value):
        """Setter for ``extentDepth`` of element ``i``.
        """
        self._ptr[i].extentDepth = value
    @property
    def extentDepth(self):
        """Getter for ``extentDepth``."""
        return self.get_extentDepth(0)
    @extentDepth.setter
    def extentDepth(self, unsigned int value):
        """Setter for ``extentDepth``."""
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
        """Getter for ``layer`` of element ``i``.
        """
        return self._ptr[i].layer
    def set_layer(self, i, unsigned int value):
        """Setter for ``layer`` of element ``i``.
        """
        self._ptr[i].layer = value
    @property
    def layer(self):
        """Getter for ``layer``."""
        return self.get_layer(0)
    @layer.setter
    def layer(self, unsigned int value):
        """Setter for ``layer``."""
        self.set_layer(0,value)
    def get_offset(self, i):
        """Getter for ``offset`` of element ``i``.
        """
        return self._ptr[i].offset
    def set_offset(self, i, unsigned long long value):
        """Setter for ``offset`` of element ``i``.
        """
        self._ptr[i].offset = value
    @property
    def offset(self):
        """Getter for ``offset``."""
        return self.get_offset(0)
    @offset.setter
    def offset(self, unsigned long long value):
        """Setter for ``offset``."""
        self.set_offset(0,value)
    def get_size(self, i):
        """Getter for ``size`` of element ``i``.
        """
        return self._ptr[i].size
    def set_size(self, i, unsigned long long value):
        """Setter for ``size`` of element ``i``.
        """
        self._ptr[i].size = value
    @property
    def size(self):
        """Getter for ``size``."""
        return self.get_size(0)
    @size.setter
    def size(self, unsigned long long value):
        """Setter for ``size``."""
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
        """Getter for ``sparseLevel`` of element ``i``.
        """
        return hipArrayMapInfo_union_1_struct_0.from_ptr(&self._ptr[i].sparseLevel)
    @property
    def sparseLevel(self):
        """Getter for ``sparseLevel``."""
        return self.get_sparseLevel(0)
    def get_miptail(self, i):
        """Getter for ``miptail`` of element ``i``.
        """
        return hipArrayMapInfo_union_1_struct_1.from_ptr(&self._ptr[i].miptail)
    @property
    def miptail(self):
        """Getter for ``miptail``."""
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
        """Getter for ``resourceType`` of element ``i``.
        """
        return hipResourceType(self._ptr[i].resourceType)
    def set_resourceType(self, i, value):
        """Setter for ``resourceType`` of element ``i``.
        """
        if not isinstance(value, hipResourceType):
            raise TypeError("'value' must be of type 'hipResourceType'")
        self._ptr[i].resourceType = value.value
    @property
    def resourceType(self):
        """Getter for ``resourceType``."""
        return self.get_resourceType(0)
    @resourceType.setter
    def resourceType(self, value):
        """Setter for ``resourceType``."""
        self.set_resourceType(0,value)
    def get_resource(self, i):
        """Getter for ``resource`` of element ``i``.
        """
        return hipArrayMapInfo_union_0.from_ptr(&self._ptr[i].resource)
    @property
    def resource(self):
        """Getter for ``resource``."""
        return self.get_resource(0)
    def get_subresourceType(self, i):
        """Getter for ``subresourceType`` of element ``i``.
        """
        return hipArraySparseSubresourceType(self._ptr[i].subresourceType)
    def set_subresourceType(self, i, value):
        """Setter for ``subresourceType`` of element ``i``.
        """
        if not isinstance(value, hipArraySparseSubresourceType):
            raise TypeError("'value' must be of type 'hipArraySparseSubresourceType'")
        self._ptr[i].subresourceType = value.value
    @property
    def subresourceType(self):
        """Getter for ``subresourceType``."""
        return self.get_subresourceType(0)
    @subresourceType.setter
    def subresourceType(self, value):
        """Setter for ``subresourceType``."""
        self.set_subresourceType(0,value)
    def get_subresource(self, i):
        """Getter for ``subresource`` of element ``i``.
        """
        return hipArrayMapInfo_union_1.from_ptr(&self._ptr[i].subresource)
    @property
    def subresource(self):
        """Getter for ``subresource``."""
        return self.get_subresource(0)
    def get_memOperationType(self, i):
        """Getter for ``memOperationType`` of element ``i``.
        """
        return hipMemOperationType(self._ptr[i].memOperationType)
    def set_memOperationType(self, i, value):
        """Setter for ``memOperationType`` of element ``i``.
        """
        if not isinstance(value, hipMemOperationType):
            raise TypeError("'value' must be of type 'hipMemOperationType'")
        self._ptr[i].memOperationType = value.value
    @property
    def memOperationType(self):
        """Getter for ``memOperationType``."""
        return self.get_memOperationType(0)
    @memOperationType.setter
    def memOperationType(self, value):
        """Setter for ``memOperationType``."""
        self.set_memOperationType(0,value)
    def get_memHandleType(self, i):
        """Getter for ``memHandleType`` of element ``i``.
        """
        return hipMemHandleType(self._ptr[i].memHandleType)
    def set_memHandleType(self, i, value):
        """Setter for ``memHandleType`` of element ``i``.
        """
        if not isinstance(value, hipMemHandleType):
            raise TypeError("'value' must be of type 'hipMemHandleType'")
        self._ptr[i].memHandleType = value.value
    @property
    def memHandleType(self):
        """Getter for ``memHandleType``."""
        return self.get_memHandleType(0)
    @memHandleType.setter
    def memHandleType(self, value):
        """Setter for ``memHandleType``."""
        self.set_memHandleType(0,value)
    def get_memHandle(self, i):
        """Getter for ``memHandle`` of element ``i``.
        """
        return hipArrayMapInfo_union_2.from_ptr(&self._ptr[i].memHandle)
    @property
    def memHandle(self):
        """Getter for ``memHandle``."""
        return self.get_memHandle(0)
    def get_offset(self, i):
        """Getter for ``offset`` of element ``i``.
        """
        return self._ptr[i].offset
    def set_offset(self, i, unsigned long long value):
        """Setter for ``offset`` of element ``i``.
        """
        self._ptr[i].offset = value
    @property
    def offset(self):
        """Getter for ``offset``."""
        return self.get_offset(0)
    @offset.setter
    def offset(self, unsigned long long value):
        """Setter for ``offset``."""
        self.set_offset(0,value)
    def get_deviceBitMask(self, i):
        """Getter for ``deviceBitMask`` of element ``i``.
        """
        return self._ptr[i].deviceBitMask
    def set_deviceBitMask(self, i, unsigned int value):
        """Setter for ``deviceBitMask`` of element ``i``.
        """
        self._ptr[i].deviceBitMask = value
    @property
    def deviceBitMask(self):
        """Getter for ``deviceBitMask``."""
        return self.get_deviceBitMask(0)
    @deviceBitMask.setter
    def deviceBitMask(self, unsigned int value):
        """Setter for ``deviceBitMask``."""
        self.set_deviceBitMask(0,value)
    def get_flags(self, i):
        """Getter for ``flags`` of element ``i``.
        """
        return self._ptr[i].flags
    def set_flags(self, i, unsigned int value):
        """Setter for ``flags`` of element ``i``.
        """
        self._ptr[i].flags = value
    @property
    def flags(self):
        """Getter for ``flags``."""
        return self.get_flags(0)
    @flags.setter
    def flags(self, unsigned int value):
        """Setter for ``flags``."""
        self.set_flags(0,value)



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
