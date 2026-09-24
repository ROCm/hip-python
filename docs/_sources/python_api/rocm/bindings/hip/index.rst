rocm.bindings.hip
=================

.. py:module:: rocm.bindings.hip


Attributes
----------

.. autoapisummary::

   rocm.bindings.hip.HIP_VERSION_MAJOR
   rocm.bindings.hip.HIP_VERSION_MINOR
   rocm.bindings.hip.HIP_VERSION_PATCH
   rocm.bindings.hip.HIP_VERSION_GITHASH
   rocm.bindings.hip.HIP_VERSION_BUILD_ID
   rocm.bindings.hip.HIP_VERSION_BUILD_NAME
   rocm.bindings.hip.HIP_VERSION
   rocm.bindings.hip.HIP_TRSA_OVERRIDE_FORMAT
   rocm.bindings.hip.HIP_TRSF_READ_AS_INTEGER
   rocm.bindings.hip.HIP_TRSF_NORMALIZED_COORDINATES
   rocm.bindings.hip.HIP_TRSF_SRGB
   rocm.bindings.hip.hipTextureType1D
   rocm.bindings.hip.hipTextureType2D
   rocm.bindings.hip.hipTextureType3D
   rocm.bindings.hip.hipTextureTypeCubemap
   rocm.bindings.hip.hipTextureType1DLayered
   rocm.bindings.hip.hipTextureType2DLayered
   rocm.bindings.hip.hipTextureTypeCubemapLayered
   rocm.bindings.hip.HIP_IMAGE_OBJECT_SIZE_DWORD
   rocm.bindings.hip.HIP_SAMPLER_OBJECT_SIZE_DWORD
   rocm.bindings.hip.HIP_SAMPLER_OBJECT_OFFSET_DWORD
   rocm.bindings.hip.HIP_TEXTURE_OBJECT_SIZE_DWORD
   rocm.bindings.hip.HIP_LAUNCH_PARAM_BUFFER_POINTER
   rocm.bindings.hip.HIP_LAUNCH_PARAM_BUFFER_SIZE
   rocm.bindings.hip.HIP_LAUNCH_PARAM_END
   rocm.bindings.hip.hipIpcMemLazyEnablePeerAccess
   rocm.bindings.hip.HIP_IPC_HANDLE_SIZE
   rocm.bindings.hip.hipStreamDefault
   rocm.bindings.hip.hipStreamNonBlocking
   rocm.bindings.hip.hipEventDefault
   rocm.bindings.hip.hipEventBlockingSync
   rocm.bindings.hip.hipEventDisableTiming
   rocm.bindings.hip.hipEventInterprocess
   rocm.bindings.hip.hipEventReleaseToDevice
   rocm.bindings.hip.hipEventReleaseToSystem
   rocm.bindings.hip.hipHostMallocDefault
   rocm.bindings.hip.hipHostMallocPortable
   rocm.bindings.hip.hipHostMallocMapped
   rocm.bindings.hip.hipHostMallocWriteCombined
   rocm.bindings.hip.hipHostMallocNumaUser
   rocm.bindings.hip.hipHostMallocCoherent
   rocm.bindings.hip.hipHostMallocNonCoherent
   rocm.bindings.hip.hipMemAttachGlobal
   rocm.bindings.hip.hipMemAttachHost
   rocm.bindings.hip.hipMemAttachSingle
   rocm.bindings.hip.hipDeviceMallocDefault
   rocm.bindings.hip.hipDeviceMallocFinegrained
   rocm.bindings.hip.hipMallocSignalMemory
   rocm.bindings.hip.hipHostRegisterDefault
   rocm.bindings.hip.hipHostRegisterPortable
   rocm.bindings.hip.hipHostRegisterMapped
   rocm.bindings.hip.hipHostRegisterIoMemory
   rocm.bindings.hip.hipExtHostRegisterCoarseGrained
   rocm.bindings.hip.hipDeviceScheduleAuto
   rocm.bindings.hip.hipDeviceScheduleSpin
   rocm.bindings.hip.hipDeviceScheduleYield
   rocm.bindings.hip.hipDeviceScheduleBlockingSync
   rocm.bindings.hip.hipDeviceScheduleMask
   rocm.bindings.hip.hipDeviceMapHost
   rocm.bindings.hip.hipDeviceLmemResizeToMax
   rocm.bindings.hip.hipArrayDefault
   rocm.bindings.hip.hipArrayLayered
   rocm.bindings.hip.hipArraySurfaceLoadStore
   rocm.bindings.hip.hipArrayCubemap
   rocm.bindings.hip.hipArrayTextureGather
   rocm.bindings.hip.hipOccupancyDefault
   rocm.bindings.hip.hipCooperativeLaunchMultiDeviceNoPreSync
   rocm.bindings.hip.hipCooperativeLaunchMultiDeviceNoPostSync
   rocm.bindings.hip.hipCpuDeviceId
   rocm.bindings.hip.hipInvalidDeviceId
   rocm.bindings.hip.hipExtAnyOrderLaunch
   rocm.bindings.hip.hipStreamWaitValueGte
   rocm.bindings.hip.hipStreamWaitValueEq
   rocm.bindings.hip.hipStreamWaitValueAnd
   rocm.bindings.hip.hipStreamWaitValueNor
   rocm.bindings.hip.hipLibraryOption
   rocm.bindings.hip.HIP_SUCCESS
   rocm.bindings.hip.HIP_ERROR_INVALID_VALUE
   rocm.bindings.hip.HIP_ERROR_NOT_INITIALIZED
   rocm.bindings.hip.HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
   rocm.bindings.hip.hipUUID
   rocm.bindings.hip.hipArray_t
   rocm.bindings.hip.hipArray_const_t
   rocm.bindings.hip.hipMipmappedArray_t
   rocm.bindings.hip.hipmipmappedArray
   rocm.bindings.hip.hipMipmappedArray_const_t
   rocm.bindings.hip.HIPresourcetype
   rocm.bindings.hip.hipResourcetype
   rocm.bindings.hip.HIPaddress_mode
   rocm.bindings.hip.HIPfilter_mode
   rocm.bindings.hip.HIP_TEXTURE_DESC
   rocm.bindings.hip.HIPresourceViewFormat
   rocm.bindings.hip.HIP_RESOURCE_DESC
   rocm.bindings.hip.HIP_RESOURCE_VIEW_DESC
   rocm.bindings.hip.hipTextureObject_t
   rocm.bindings.hip.hipSurfaceObject_t
   rocm.bindings.hip.hipCtx_t
   rocm.bindings.hip.hipExecutionCtx_t
   rocm.bindings.hip.hipDevResourceDesc_t
   rocm.bindings.hip.hipDevResource
   rocm.bindings.hip.hipDevSmResourceGroupParams
   rocm.bindings.hip.hipStream_t
   rocm.bindings.hip.hipIpcMemHandle_t
   rocm.bindings.hip.hipIpcEventHandle_t
   rocm.bindings.hip.hipMemFabricHandle_t
   rocm.bindings.hip.hipModule_t
   rocm.bindings.hip.hipFunction_t
   rocm.bindings.hip.hipLinkState_t
   rocm.bindings.hip.hipLibrary_t
   rocm.bindings.hip.hipKernel_t
   rocm.bindings.hip.hipMemPool_t
   rocm.bindings.hip.hipEvent_t
   rocm.bindings.hip.hipStreamBatchMemOpParams
   rocm.bindings.hip.hipLaunchParams
   rocm.bindings.hip.hipFunctionLaunchParams
   rocm.bindings.hip.hipExternalMemoryHandleType
   rocm.bindings.hip.hipExternalMemoryHandleDesc
   rocm.bindings.hip.hipExternalMemoryBufferDesc
   rocm.bindings.hip.hipExternalMemoryMipmappedArrayDesc
   rocm.bindings.hip.hipExternalSemaphoreHandleType
   rocm.bindings.hip.hipExternalSemaphoreHandleDesc
   rocm.bindings.hip.hipExternalSemaphoreSignalParams
   rocm.bindings.hip.hipExternalSemaphoreWaitParams
   rocm.bindings.hip.hipGraphicsResource
   rocm.bindings.hip.hipGraphicsResource_t
   rocm.bindings.hip.hipGraph_t
   rocm.bindings.hip.hipGraphNode_t
   rocm.bindings.hip.hipGraphExec_t
   rocm.bindings.hip.hipUserObject_t
   rocm.bindings.hip.hipMemGenericAllocationHandle_t
   rocm.bindings.hip.hipLaunchAttribute
   rocm.bindings.hip.hipLaunchConfig_t
   rocm.bindings.hip.HIP_LAUNCH_CONFIG


Classes
-------

.. autoapisummary::

   rocm.bindings.hip.hipJitOption
   rocm.bindings.hip.hipJitInputType
   rocm.bindings.hip.hipJitCacheMode
   rocm.bindings.hip.hipJitFallback
   rocm.bindings.hip.hipLibraryOption_e
   rocm.bindings.hip.hipDeviceArch_t
   rocm.bindings.hip.hipUUID_t
   rocm.bindings.hip.hipDeviceProp_t
   rocm.bindings.hip.hipMemoryType
   rocm.bindings.hip.hipPointerAttribute_t
   rocm.bindings.hip.hipError_t
   rocm.bindings.hip.hipDeviceAttribute_t
   rocm.bindings.hip.hipDriverProcAddressQueryResult
   rocm.bindings.hip.hipComputeMode
   rocm.bindings.hip.hipFlushGPUDirectRDMAWritesOptions
   rocm.bindings.hip.hipGPUDirectRDMAWritesOrdering
   rocm.bindings.hip.hipChannelFormatKind
   rocm.bindings.hip.hipChannelFormatDesc
   rocm.bindings.hip.hipArray
   rocm.bindings.hip.hipArray_Format
   rocm.bindings.hip.HIP_ARRAY_DESCRIPTOR
   rocm.bindings.hip.HIP_ARRAY3D_DESCRIPTOR
   rocm.bindings.hip.hip_Memcpy2D
   rocm.bindings.hip.hipMipmappedArray
   rocm.bindings.hip.hipResourceType
   rocm.bindings.hip.HIPresourcetype_enum
   rocm.bindings.hip.HIPaddress_mode_enum
   rocm.bindings.hip.HIPfilter_mode_enum
   rocm.bindings.hip.HIP_TEXTURE_DESC_st
   rocm.bindings.hip.hipResourceViewFormat
   rocm.bindings.hip.HIPresourceViewFormat_enum
   rocm.bindings.hip.hipResourceDesc_union_0_struct_0
   rocm.bindings.hip.hipResourceDesc_union_0_struct_1
   rocm.bindings.hip.hipResourceDesc_union_0_struct_2
   rocm.bindings.hip.hipResourceDesc_union_0_struct_3
   rocm.bindings.hip.hipResourceDesc_union_0
   rocm.bindings.hip.hipResourceDesc
   rocm.bindings.hip.HIP_RESOURCE_DESC_st_union_0_struct_0
   rocm.bindings.hip.HIP_RESOURCE_DESC_st_union_0_struct_1
   rocm.bindings.hip.HIP_RESOURCE_DESC_st_union_0_struct_2
   rocm.bindings.hip.HIP_RESOURCE_DESC_st_union_0_struct_3
   rocm.bindings.hip.HIP_RESOURCE_DESC_st_union_0_struct_4
   rocm.bindings.hip.HIP_RESOURCE_DESC_st_union_0
   rocm.bindings.hip.HIP_RESOURCE_DESC_st
   rocm.bindings.hip.hipResourceViewDesc
   rocm.bindings.hip.HIP_RESOURCE_VIEW_DESC_st
   rocm.bindings.hip.hipMemcpyKind
   rocm.bindings.hip.hipPitchedPtr
   rocm.bindings.hip.hipExtent
   rocm.bindings.hip.hipPos
   rocm.bindings.hip.hipMemcpy3DParms
   rocm.bindings.hip.HIP_MEMCPY3D
   rocm.bindings.hip.hipMemLocationType
   rocm.bindings.hip.hipMemLocation
   rocm.bindings.hip.hipMemcpyFlags
   rocm.bindings.hip.hipMemcpySrcAccessOrder
   rocm.bindings.hip.hipMemcpyAttributes
   rocm.bindings.hip.hipMemcpy3DOperandType
   rocm.bindings.hip.hipOffset3D
   rocm.bindings.hip.hipMemcpy3DOperand_union_0_struct_0
   rocm.bindings.hip.hipMemcpy3DOperand_union_0_struct_1
   rocm.bindings.hip.hipMemcpy3DOperand_union_0
   rocm.bindings.hip.hipMemcpy3DOperand
   rocm.bindings.hip.hipMemcpy3DBatchOp
   rocm.bindings.hip.hipMemcpy3DPeerParms
   rocm.bindings.hip.hipFunction_attribute
   rocm.bindings.hip.hipPointer_attribute
   rocm.bindings.hip.uchar1
   rocm.bindings.hip.uchar2
   rocm.bindings.hip.uchar3
   rocm.bindings.hip.uchar4
   rocm.bindings.hip.char1
   rocm.bindings.hip.char2
   rocm.bindings.hip.char3
   rocm.bindings.hip.char4
   rocm.bindings.hip.ushort1
   rocm.bindings.hip.ushort2
   rocm.bindings.hip.ushort3
   rocm.bindings.hip.ushort4
   rocm.bindings.hip.short1
   rocm.bindings.hip.short2
   rocm.bindings.hip.short3
   rocm.bindings.hip.short4
   rocm.bindings.hip.uint1
   rocm.bindings.hip.uint2
   rocm.bindings.hip.uint3
   rocm.bindings.hip.uint4
   rocm.bindings.hip.int1
   rocm.bindings.hip.int2
   rocm.bindings.hip.int3
   rocm.bindings.hip.int4
   rocm.bindings.hip.ulong1
   rocm.bindings.hip.ulong2
   rocm.bindings.hip.ulong3
   rocm.bindings.hip.ulong4
   rocm.bindings.hip.long1
   rocm.bindings.hip.long2
   rocm.bindings.hip.long3
   rocm.bindings.hip.long4
   rocm.bindings.hip.ulonglong1
   rocm.bindings.hip.ulonglong2
   rocm.bindings.hip.ulonglong3
   rocm.bindings.hip.ulonglong4
   rocm.bindings.hip.longlong1
   rocm.bindings.hip.longlong2
   rocm.bindings.hip.longlong3
   rocm.bindings.hip.longlong4
   rocm.bindings.hip.float1
   rocm.bindings.hip.float2
   rocm.bindings.hip.float3
   rocm.bindings.hip.float4
   rocm.bindings.hip.double1
   rocm.bindings.hip.double2
   rocm.bindings.hip.double3
   rocm.bindings.hip.double4
   rocm.bindings.hip.hipTextureAddressMode
   rocm.bindings.hip.hipTextureFilterMode
   rocm.bindings.hip.hipTextureReadMode
   rocm.bindings.hip.textureReference
   rocm.bindings.hip.hipTextureDesc
   rocm.bindings.hip.surfaceReference
   rocm.bindings.hip.hipSurfaceBoundaryMode
   rocm.bindings.hip.ihipCtx_t
   rocm.bindings.hip.ihipExecutionCtx_t
   rocm.bindings.hip.ihipDevResourceDesc_t
   rocm.bindings.hip.hipDevResourceType
   rocm.bindings.hip.hipDevSmResourceGroup_flags
   rocm.bindings.hip.hipDevSmResourceSplitByCount_flags
   rocm.bindings.hip.hipDevWorkqueueConfigScope
   rocm.bindings.hip.hipDevSmResource
   rocm.bindings.hip.hipDevWorkqueueConfigResource
   rocm.bindings.hip.hipDevWorkqueueResource
   rocm.bindings.hip.hipDevResource_st_union_0
   rocm.bindings.hip.hipDevResource_st
   rocm.bindings.hip.hipDevSmResourceGroupParams_st
   rocm.bindings.hip.hipDeviceP2PAttr
   rocm.bindings.hip.hipDriverEntryPointQueryResult
   rocm.bindings.hip.ihipStream_t
   rocm.bindings.hip.hipIpcMemHandle_st
   rocm.bindings.hip.hipIpcEventHandle_st
   rocm.bindings.hip.hipMemFabricHandle_st
   rocm.bindings.hip.ihipModule_t
   rocm.bindings.hip.ihipModuleSymbol_t
   rocm.bindings.hip.ihipLinkState_t
   rocm.bindings.hip.ihipLibrary_t
   rocm.bindings.hip.ihipKernel_t
   rocm.bindings.hip.ihipMemPoolHandle_t
   rocm.bindings.hip.hipFuncAttributes
   rocm.bindings.hip.ihipEvent_t
   rocm.bindings.hip.hipLimit_t
   rocm.bindings.hip.hipStreamBatchMemOpType
   rocm.bindings.hip.hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t_union_0
   rocm.bindings.hip.hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t
   rocm.bindings.hip.hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t_union_0
   rocm.bindings.hip.hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t
   rocm.bindings.hip.hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t
   rocm.bindings.hip.hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t
   rocm.bindings.hip.hipStreamBatchMemOpParams_union
   rocm.bindings.hip.hipBatchMemOpNodeParams
   rocm.bindings.hip.hipMemoryAdvise
   rocm.bindings.hip.hipMemRangeCoherencyMode
   rocm.bindings.hip.hipMemRangeAttribute
   rocm.bindings.hip.hipMemPoolAttr
   rocm.bindings.hip.hipMemAccessFlags
   rocm.bindings.hip.hipMemAccessDesc
   rocm.bindings.hip.hipMemAllocationType
   rocm.bindings.hip.hipMemAllocationHandleType
   rocm.bindings.hip.hipMemPoolProps
   rocm.bindings.hip.hipMemPoolPtrExportData
   rocm.bindings.hip.hipFuncAttribute
   rocm.bindings.hip.hipFuncCache_t
   rocm.bindings.hip.hipSharedMemConfig
   rocm.bindings.hip.dim3
   rocm.bindings.hip.hipLaunchParams_t
   rocm.bindings.hip.hipFunctionLaunchParams_t
   rocm.bindings.hip.hipExternalMemoryHandleType_enum
   rocm.bindings.hip.hipExternalMemoryHandleDesc_st_union_0_struct_0
   rocm.bindings.hip.hipExternalMemoryHandleDesc_st_union_0
   rocm.bindings.hip.hipExternalMemoryHandleDesc_st
   rocm.bindings.hip.hipExternalMemoryBufferDesc_st
   rocm.bindings.hip.hipExternalMemoryMipmappedArrayDesc_st
   rocm.bindings.hip.hipExternalSemaphoreHandleType_enum
   rocm.bindings.hip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0
   rocm.bindings.hip.hipExternalSemaphoreHandleDesc_st_union_0
   rocm.bindings.hip.hipExternalSemaphoreHandleDesc_st
   rocm.bindings.hip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0
   rocm.bindings.hip.hipExternalSemaphoreSignalParams_st_struct_0_union_0
   rocm.bindings.hip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1
   rocm.bindings.hip.hipExternalSemaphoreSignalParams_st_struct_0
   rocm.bindings.hip.hipExternalSemaphoreSignalParams_st
   rocm.bindings.hip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0
   rocm.bindings.hip.hipExternalSemaphoreWaitParams_st_struct_0_union_0
   rocm.bindings.hip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1
   rocm.bindings.hip.hipExternalSemaphoreWaitParams_st_struct_0
   rocm.bindings.hip.hipExternalSemaphoreWaitParams_st
   rocm.bindings.hip.hipGraphicsRegisterFlags
   rocm.bindings.hip.ihipGraph
   rocm.bindings.hip.hipGraphNode
   rocm.bindings.hip.hipGraphExec
   rocm.bindings.hip.hipUserObject
   rocm.bindings.hip.hipGraphNodeType
   rocm.bindings.hip.hipHostFn_t
   rocm.bindings.hip.hipHostNodeParams
   rocm.bindings.hip.hipKernelNodeParams
   rocm.bindings.hip.hipMemsetParams
   rocm.bindings.hip.hipMemAllocNodeParams
   rocm.bindings.hip.hipAccessProperty
   rocm.bindings.hip.hipAccessPolicyWindow
   rocm.bindings.hip.hipLaunchMemSyncDomainMap
   rocm.bindings.hip.hipLaunchMemSyncDomain
   rocm.bindings.hip.hipSynchronizationPolicy
   rocm.bindings.hip.hipClusterSchedulingPolicy
   rocm.bindings.hip.hipExtDynDataPrefetchTemporal
   rocm.bindings.hip.hipExtDynDataPrefetchRegion
   rocm.bindings.hip.hipExtDynDataPrefetchConfig
   rocm.bindings.hip.hipLaunchAttributeID
   rocm.bindings.hip.hipLaunchAttributeValue_struct_0
   rocm.bindings.hip.hipLaunchAttributeValue
   rocm.bindings.hip.hipGraphExecUpdateResult
   rocm.bindings.hip.hipStreamCaptureMode
   rocm.bindings.hip.hipStreamCaptureStatus
   rocm.bindings.hip.hipStreamUpdateCaptureDependenciesFlags
   rocm.bindings.hip.hipGraphMemAttributeType
   rocm.bindings.hip.hipUserObjectFlags
   rocm.bindings.hip.hipUserObjectRetainFlags
   rocm.bindings.hip.hipGraphInstantiateFlags
   rocm.bindings.hip.hipGraphDebugDotFlags
   rocm.bindings.hip.hipGraphInstantiateResult
   rocm.bindings.hip.hipGraphInstantiateParams
   rocm.bindings.hip.hipMemAllocationProp_union_0
   rocm.bindings.hip.hipMemAllocationProp_struct_0
   rocm.bindings.hip.hipMemAllocationProp
   rocm.bindings.hip.hipExternalSemaphoreSignalNodeParams
   rocm.bindings.hip.hipExternalSemaphoreWaitNodeParams
   rocm.bindings.hip.ihipMemGenericAllocationHandle
   rocm.bindings.hip.hipMemAllocationGranularity_flags
   rocm.bindings.hip.hipMemHandleType
   rocm.bindings.hip.hipMemOperationType
   rocm.bindings.hip.hipArraySparseSubresourceType
   rocm.bindings.hip.hipArrayMapInfo_union_0
   rocm.bindings.hip.hipArrayMapInfo_union_1_struct_0
   rocm.bindings.hip.hipArrayMapInfo_union_1_struct_1
   rocm.bindings.hip.hipArrayMapInfo_union_1
   rocm.bindings.hip.hipArrayMapInfo_union_2
   rocm.bindings.hip.hipArrayMapInfo
   rocm.bindings.hip.hipMemcpyNodeParams
   rocm.bindings.hip.hipChildGraphNodeParams
   rocm.bindings.hip.hipEventWaitNodeParams
   rocm.bindings.hip.hipEventRecordNodeParams
   rocm.bindings.hip.hipMemFreeNodeParams
   rocm.bindings.hip.hipGraphNodeParams_union_0
   rocm.bindings.hip.hipGraphNodeParams
   rocm.bindings.hip.hipGraphDependencyType
   rocm.bindings.hip.hipGraphEdgeData
   rocm.bindings.hip.hipLaunchAttribute_st_union_0
   rocm.bindings.hip.hipLaunchAttribute_st
   rocm.bindings.hip.hipLaunchConfig_st
   rocm.bindings.hip.HIP_LAUNCH_CONFIG_st
   rocm.bindings.hip.hipArrayMemoryRequirements
   rocm.bindings.hip.hipMemRangeHandleType
   rocm.bindings.hip.hipMemRangeFlags
   rocm.bindings.hip.hipStreamCallback_t
   rocm.bindings.hip.hipDataType
   rocm.bindings.hip.hipLibraryPropertyType


Functions
---------

.. autoapisummary::

   rocm.bindings.hip.has_symbol
   rocm.bindings.hip.hipCreateChannelDesc
   rocm.bindings.hip.hipInit
   rocm.bindings.hip.hipDriverGetVersion
   rocm.bindings.hip.hipRuntimeGetVersion
   rocm.bindings.hip.hipDeviceGet
   rocm.bindings.hip.hipDeviceComputeCapability
   rocm.bindings.hip.hipDeviceGetName
   rocm.bindings.hip.hipDeviceGetUuid
   rocm.bindings.hip.hipDeviceGetP2PAttribute
   rocm.bindings.hip.hipDeviceGetPCIBusId
   rocm.bindings.hip.hipDeviceGetByPCIBusId
   rocm.bindings.hip.hipDeviceTotalMem
   rocm.bindings.hip.hipDeviceSynchronize
   rocm.bindings.hip.hipDeviceReset
   rocm.bindings.hip.hipSetDevice
   rocm.bindings.hip.hipSetValidDevices
   rocm.bindings.hip.hipGetDevice
   rocm.bindings.hip.hipGetDeviceCount
   rocm.bindings.hip.hipDeviceGetAttribute
   rocm.bindings.hip.hipDeviceGetDefaultMemPool
   rocm.bindings.hip.hipDeviceSetMemPool
   rocm.bindings.hip.hipDeviceGetMemPool
   rocm.bindings.hip.hipGetDeviceProperties
   rocm.bindings.hip.hipDeviceGetTexture1DLinearMaxWidth
   rocm.bindings.hip.hipDeviceSetCacheConfig
   rocm.bindings.hip.hipDeviceGetCacheConfig
   rocm.bindings.hip.hipDeviceGetLimit
   rocm.bindings.hip.hipDeviceSetLimit
   rocm.bindings.hip.hipDeviceGetSharedMemConfig
   rocm.bindings.hip.hipGetDeviceFlags
   rocm.bindings.hip.hipDeviceSetSharedMemConfig
   rocm.bindings.hip.hipSetDeviceFlags
   rocm.bindings.hip.hipChooseDevice
   rocm.bindings.hip.hipExtGetLinkTypeAndHopCount
   rocm.bindings.hip.hipIpcGetMemHandle
   rocm.bindings.hip.hipIpcOpenMemHandle
   rocm.bindings.hip.hipIpcCloseMemHandle
   rocm.bindings.hip.hipIpcGetEventHandle
   rocm.bindings.hip.hipIpcOpenEventHandle
   rocm.bindings.hip.hipFuncSetAttribute
   rocm.bindings.hip.hipKernelSetAttribute
   rocm.bindings.hip.hipKernelGetFunction
   rocm.bindings.hip.hipFuncSetCacheConfig
   rocm.bindings.hip.hipFuncSetSharedMemConfig
   rocm.bindings.hip.hipGetLastError
   rocm.bindings.hip.hipExtGetLastError
   rocm.bindings.hip.hipPeekAtLastError
   rocm.bindings.hip.hipGetErrorName
   rocm.bindings.hip.hipGetErrorString
   rocm.bindings.hip.hipDrvGetErrorName
   rocm.bindings.hip.hipDrvGetErrorString
   rocm.bindings.hip.hipStreamCreate
   rocm.bindings.hip.hipStreamCreateWithFlags
   rocm.bindings.hip.hipStreamCreateWithPriority
   rocm.bindings.hip.hipDeviceGetStreamPriorityRange
   rocm.bindings.hip.hipStreamDestroy
   rocm.bindings.hip.hipStreamQuery
   rocm.bindings.hip.hipStreamSynchronize
   rocm.bindings.hip.hipStreamWaitEvent
   rocm.bindings.hip.hipStreamGetFlags
   rocm.bindings.hip.hipStreamGetId
   rocm.bindings.hip.hipStreamGetPriority
   rocm.bindings.hip.hipStreamGetDevice
   rocm.bindings.hip.hipExtStreamCreateWithCUMask
   rocm.bindings.hip.hipExtStreamGetCUMask
   rocm.bindings.hip.hipStreamAddCallback
   rocm.bindings.hip.hipStreamSetAttribute
   rocm.bindings.hip.hipStreamGetAttribute
   rocm.bindings.hip.hipStreamCopyAttributes
   rocm.bindings.hip.hipStreamWaitValue32
   rocm.bindings.hip.hipStreamWaitValue64
   rocm.bindings.hip.hipStreamWriteValue32
   rocm.bindings.hip.hipStreamWriteValue64
   rocm.bindings.hip.hipStreamBatchMemOp
   rocm.bindings.hip.hipGraphAddBatchMemOpNode
   rocm.bindings.hip.hipGraphBatchMemOpNodeGetParams
   rocm.bindings.hip.hipGraphBatchMemOpNodeSetParams
   rocm.bindings.hip.hipGraphExecBatchMemOpNodeSetParams
   rocm.bindings.hip.hipEventCreateWithFlags
   rocm.bindings.hip.hipEventCreate
   rocm.bindings.hip.hipEventRecordWithFlags
   rocm.bindings.hip.hipEventRecord
   rocm.bindings.hip.hipEventDestroy
   rocm.bindings.hip.hipEventSynchronize
   rocm.bindings.hip.hipEventElapsedTime
   rocm.bindings.hip.hipEventQuery
   rocm.bindings.hip.hipPointerSetAttribute
   rocm.bindings.hip.hipPointerGetAttributes
   rocm.bindings.hip.hipPointerGetAttribute
   rocm.bindings.hip.hipDrvPointerGetAttributes
   rocm.bindings.hip.hipImportExternalSemaphore
   rocm.bindings.hip.hipSignalExternalSemaphoresAsync
   rocm.bindings.hip.hipWaitExternalSemaphoresAsync
   rocm.bindings.hip.hipDestroyExternalSemaphore
   rocm.bindings.hip.hipImportExternalMemory
   rocm.bindings.hip.hipExternalMemoryGetMappedBuffer
   rocm.bindings.hip.hipDestroyExternalMemory
   rocm.bindings.hip.hipExternalMemoryGetMappedMipmappedArray
   rocm.bindings.hip.hipMalloc
   rocm.bindings.hip.hipExtMallocWithFlags
   rocm.bindings.hip.hipMallocHost
   rocm.bindings.hip.hipMemAllocHost
   rocm.bindings.hip.hipHostMalloc
   rocm.bindings.hip.hipMallocManaged
   rocm.bindings.hip.hipMemPrefetchAsync
   rocm.bindings.hip.hipMemPrefetchAsync_v2
   rocm.bindings.hip.hipMemPrefetchBatchAsync
   rocm.bindings.hip.hipMemDiscardBatchAsync
   rocm.bindings.hip.hipDrvMemDiscardBatchAsync
   rocm.bindings.hip.hipMemDiscardAndPrefetchBatchAsync
   rocm.bindings.hip.hipDrvMemDiscardAndPrefetchBatchAsync
   rocm.bindings.hip.hipMemAdvise
   rocm.bindings.hip.hipMemAdvise_v2
   rocm.bindings.hip.hipMemRangeGetAttribute
   rocm.bindings.hip.hipMemRangeGetAttributes
   rocm.bindings.hip.hipStreamAttachMemAsync
   rocm.bindings.hip.hipMallocAsync
   rocm.bindings.hip.hipFreeAsync
   rocm.bindings.hip.hipMemPoolTrimTo
   rocm.bindings.hip.hipMemPoolSetAttribute
   rocm.bindings.hip.hipMemPoolGetAttribute
   rocm.bindings.hip.hipMemPoolSetAccess
   rocm.bindings.hip.hipMemPoolGetAccess
   rocm.bindings.hip.hipMemPoolCreate
   rocm.bindings.hip.hipMemPoolDestroy
   rocm.bindings.hip.hipMallocFromPoolAsync
   rocm.bindings.hip.hipMemPoolExportToShareableHandle
   rocm.bindings.hip.hipMemPoolImportFromShareableHandle
   rocm.bindings.hip.hipMemPoolExportPointer
   rocm.bindings.hip.hipMemPoolImportPointer
   rocm.bindings.hip.hipMemSetMemPool
   rocm.bindings.hip.hipMemGetMemPool
   rocm.bindings.hip.hipMemGetDefaultMemPool
   rocm.bindings.hip.hipHostAlloc
   rocm.bindings.hip.hipHostGetDevicePointer
   rocm.bindings.hip.hipHostGetFlags
   rocm.bindings.hip.hipHostRegister
   rocm.bindings.hip.hipHostUnregister
   rocm.bindings.hip.hipMallocPitch
   rocm.bindings.hip.hipMemAllocPitch
   rocm.bindings.hip.hipFree
   rocm.bindings.hip.hipFreeHost
   rocm.bindings.hip.hipHostFree
   rocm.bindings.hip.hipMemcpy
   rocm.bindings.hip.hipMemcpyWithStream
   rocm.bindings.hip.hipMemcpyHtoD
   rocm.bindings.hip.hipMemcpyDtoH
   rocm.bindings.hip.hipMemcpyDtoD
   rocm.bindings.hip.hipMemcpyAtoD
   rocm.bindings.hip.hipMemcpyDtoA
   rocm.bindings.hip.hipMemcpyAtoA
   rocm.bindings.hip.hipMemcpyHtoDAsync
   rocm.bindings.hip.hipMemcpyDtoHAsync
   rocm.bindings.hip.hipMemcpyDtoDAsync
   rocm.bindings.hip.hipMemcpyAtoHAsync
   rocm.bindings.hip.hipMemcpyHtoAAsync
   rocm.bindings.hip.hipModuleGetGlobal
   rocm.bindings.hip.hipGetSymbolAddress
   rocm.bindings.hip.hipGetSymbolSize
   rocm.bindings.hip.hipGetProcAddress
   rocm.bindings.hip.hipMemcpyToSymbol
   rocm.bindings.hip.hipMemcpyToSymbolAsync
   rocm.bindings.hip.hipMemcpyFromSymbol
   rocm.bindings.hip.hipMemcpyFromSymbolAsync
   rocm.bindings.hip.hipMemcpyAsync
   rocm.bindings.hip.hipMemset
   rocm.bindings.hip.hipMemsetD8
   rocm.bindings.hip.hipMemsetD8Async
   rocm.bindings.hip.hipMemsetD16
   rocm.bindings.hip.hipMemsetD16Async
   rocm.bindings.hip.hipMemsetD32
   rocm.bindings.hip.hipMemsetAsync
   rocm.bindings.hip.hipMemsetD32Async
   rocm.bindings.hip.hipMemset2D
   rocm.bindings.hip.hipMemset2DAsync
   rocm.bindings.hip.hipMemset3D
   rocm.bindings.hip.hipMemset3DAsync
   rocm.bindings.hip.hipMemsetD2D8
   rocm.bindings.hip.hipMemsetD2D8Async
   rocm.bindings.hip.hipMemsetD2D16
   rocm.bindings.hip.hipMemsetD2D16Async
   rocm.bindings.hip.hipMemsetD2D32
   rocm.bindings.hip.hipMemsetD2D32Async
   rocm.bindings.hip.hipMemGetInfo
   rocm.bindings.hip.hipMemPtrGetInfo
   rocm.bindings.hip.hipMallocArray
   rocm.bindings.hip.hipArrayCreate
   rocm.bindings.hip.hipArrayDestroy
   rocm.bindings.hip.hipArray3DCreate
   rocm.bindings.hip.hipMalloc3D
   rocm.bindings.hip.hipFreeArray
   rocm.bindings.hip.hipMalloc3DArray
   rocm.bindings.hip.hipArrayGetInfo
   rocm.bindings.hip.hipArrayGetDescriptor
   rocm.bindings.hip.hipArray3DGetDescriptor
   rocm.bindings.hip.hipMemcpy2D
   rocm.bindings.hip.hipMemcpyParam2D
   rocm.bindings.hip.hipMemcpyParam2DAsync
   rocm.bindings.hip.hipMemcpy2DAsync
   rocm.bindings.hip.hipMemcpy2DToArray
   rocm.bindings.hip.hipMemcpy2DToArrayAsync
   rocm.bindings.hip.hipMemcpy2DArrayToArray
   rocm.bindings.hip.hipMemcpyToArray
   rocm.bindings.hip.hipMemcpyFromArray
   rocm.bindings.hip.hipMemcpy2DFromArray
   rocm.bindings.hip.hipMemcpy2DFromArrayAsync
   rocm.bindings.hip.hipMemcpyAtoH
   rocm.bindings.hip.hipMemcpyHtoA
   rocm.bindings.hip.hipMemcpy3D
   rocm.bindings.hip.hipMemcpy3DAsync
   rocm.bindings.hip.hipDrvMemcpy3D
   rocm.bindings.hip.hipDrvMemcpy3DAsync
   rocm.bindings.hip.hipMemGetAddressRange
   rocm.bindings.hip.hipMemcpyBatchAsync
   rocm.bindings.hip.hipMemcpy3DBatchAsync
   rocm.bindings.hip.hipMemcpy3DPeer
   rocm.bindings.hip.hipMemcpy3DPeerAsync
   rocm.bindings.hip.hipMipmappedArrayGetMemoryRequirements
   rocm.bindings.hip.hipDeviceCanAccessPeer
   rocm.bindings.hip.hipDeviceEnablePeerAccess
   rocm.bindings.hip.hipDeviceDisablePeerAccess
   rocm.bindings.hip.hipMemcpyPeer
   rocm.bindings.hip.hipMemcpyPeerAsync
   rocm.bindings.hip.hipDeviceGetDevResource
   rocm.bindings.hip.hipDevSmResourceSplitByCount
   rocm.bindings.hip.hipDevSmResourceSplit
   rocm.bindings.hip.hipDevResourceGenerateDesc
   rocm.bindings.hip.hipGreenCtxCreate
   rocm.bindings.hip.hipExecutionCtxDestroy
   rocm.bindings.hip.hipDeviceGetExecutionCtx
   rocm.bindings.hip.hipExecutionCtxStreamCreate
   rocm.bindings.hip.hipExecutionCtxGetDevResource
   rocm.bindings.hip.hipExecutionCtxGetDevice
   rocm.bindings.hip.hipExecutionCtxGetId
   rocm.bindings.hip.hipStreamGetDevResource
   rocm.bindings.hip.hipExecutionCtxRecordEvent
   rocm.bindings.hip.hipExecutionCtxSynchronize
   rocm.bindings.hip.hipExecutionCtxWaitEvent
   rocm.bindings.hip.hipCtxCreate
   rocm.bindings.hip.hipCtxDestroy
   rocm.bindings.hip.hipCtxPopCurrent
   rocm.bindings.hip.hipCtxPushCurrent
   rocm.bindings.hip.hipCtxSetCurrent
   rocm.bindings.hip.hipCtxGetCurrent
   rocm.bindings.hip.hipCtxGetDevice
   rocm.bindings.hip.hipCtxGetApiVersion
   rocm.bindings.hip.hipCtxGetCacheConfig
   rocm.bindings.hip.hipCtxSetCacheConfig
   rocm.bindings.hip.hipCtxSetSharedMemConfig
   rocm.bindings.hip.hipCtxGetSharedMemConfig
   rocm.bindings.hip.hipCtxSynchronize
   rocm.bindings.hip.hipCtxGetFlags
   rocm.bindings.hip.hipCtxEnablePeerAccess
   rocm.bindings.hip.hipCtxDisablePeerAccess
   rocm.bindings.hip.hipDevicePrimaryCtxGetState
   rocm.bindings.hip.hipDevicePrimaryCtxRelease
   rocm.bindings.hip.hipDevicePrimaryCtxRetain
   rocm.bindings.hip.hipDevicePrimaryCtxReset
   rocm.bindings.hip.hipDevicePrimaryCtxSetFlags
   rocm.bindings.hip.hipModuleLoadFatBinary
   rocm.bindings.hip.hipModuleLoad
   rocm.bindings.hip.hipModuleUnload
   rocm.bindings.hip.hipModuleGetFunction
   rocm.bindings.hip.hipModuleGetFunctionCount
   rocm.bindings.hip.hipKernelGetAttribute
   rocm.bindings.hip.hipLibraryLoadData
   rocm.bindings.hip.hipLibraryLoadFromFile
   rocm.bindings.hip.hipLibraryUnload
   rocm.bindings.hip.hipLibraryGetKernel
   rocm.bindings.hip.hipLibraryGetKernelCount
   rocm.bindings.hip.hipLibraryGetGlobal
   rocm.bindings.hip.hipLibraryGetManaged
   rocm.bindings.hip.hipLibraryEnumerateKernels
   rocm.bindings.hip.hipKernelGetLibrary
   rocm.bindings.hip.hipKernelGetName
   rocm.bindings.hip.hipKernelGetParamInfo
   rocm.bindings.hip.hipFuncGetAttributes
   rocm.bindings.hip.hipFuncGetAttribute
   rocm.bindings.hip.hipGetFuncBySymbol
   rocm.bindings.hip.hipGetDriverEntryPoint
   rocm.bindings.hip.hipModuleGetTexRef
   rocm.bindings.hip.hipModuleLoadData
   rocm.bindings.hip.hipModuleLoadDataEx
   rocm.bindings.hip.hipLinkAddData
   rocm.bindings.hip.hipLinkAddFile
   rocm.bindings.hip.hipLinkComplete
   rocm.bindings.hip.hipLinkCreate
   rocm.bindings.hip.hipLinkDestroy
   rocm.bindings.hip.hipModuleLaunchKernel
   rocm.bindings.hip.hipModuleLaunchCooperativeKernel
   rocm.bindings.hip.hipModuleLaunchCooperativeKernelMultiDevice
   rocm.bindings.hip.hipLaunchCooperativeKernel
   rocm.bindings.hip.hipLaunchCooperativeKernelMultiDevice
   rocm.bindings.hip.hipExtLaunchMultiKernelMultiDevice
   rocm.bindings.hip.hipLaunchKernelExC
   rocm.bindings.hip.hipDrvLaunchKernelEx
   rocm.bindings.hip.hipMemGetHandleForAddressRange
   rocm.bindings.hip.hipModuleOccupancyMaxPotentialBlockSize
   rocm.bindings.hip.hipModuleOccupancyMaxPotentialBlockSizeWithFlags
   rocm.bindings.hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
   rocm.bindings.hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
   rocm.bindings.hip.hipOccupancyMaxActiveBlocksPerMultiprocessor
   rocm.bindings.hip.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
   rocm.bindings.hip.hipOccupancyMaxPotentialBlockSize
   rocm.bindings.hip.hipOccupancyAvailableDynamicSMemPerBlock
   rocm.bindings.hip.hipOccupancyMaxActiveClusters
   rocm.bindings.hip.hipOccupancyMaxPotentialClusterSize
   rocm.bindings.hip.hipProfilerStart
   rocm.bindings.hip.hipProfilerStop
   rocm.bindings.hip.hipConfigureCall
   rocm.bindings.hip.hipSetupArgument
   rocm.bindings.hip.hipLaunchByPtr
   rocm.bindings.hip.hipLaunchKernel
   rocm.bindings.hip.hipLaunchHostFunc
   rocm.bindings.hip.hipDrvMemcpy2DUnaligned
   rocm.bindings.hip.hipExtLaunchKernel
   rocm.bindings.hip.hipCreateTextureObject
   rocm.bindings.hip.hipDestroyTextureObject
   rocm.bindings.hip.hipGetChannelDesc
   rocm.bindings.hip.hipGetTextureObjectResourceDesc
   rocm.bindings.hip.hipGetTextureObjectResourceViewDesc
   rocm.bindings.hip.hipGetTextureObjectTextureDesc
   rocm.bindings.hip.hipTexObjectCreate
   rocm.bindings.hip.hipTexObjectDestroy
   rocm.bindings.hip.hipTexObjectGetResourceDesc
   rocm.bindings.hip.hipTexObjectGetResourceViewDesc
   rocm.bindings.hip.hipTexObjectGetTextureDesc
   rocm.bindings.hip.hipMallocMipmappedArray
   rocm.bindings.hip.hipFreeMipmappedArray
   rocm.bindings.hip.hipGetMipmappedArrayLevel
   rocm.bindings.hip.hipMipmappedArrayCreate
   rocm.bindings.hip.hipMipmappedArrayDestroy
   rocm.bindings.hip.hipMipmappedArrayGetLevel
   rocm.bindings.hip.hipBindTextureToMipmappedArray
   rocm.bindings.hip.hipGetTextureReference
   rocm.bindings.hip.hipTexRefGetBorderColor
   rocm.bindings.hip.hipTexRefGetArray
   rocm.bindings.hip.hipTexRefSetAddressMode
   rocm.bindings.hip.hipTexRefSetArray
   rocm.bindings.hip.hipTexRefSetFilterMode
   rocm.bindings.hip.hipTexRefSetFlags
   rocm.bindings.hip.hipTexRefSetFormat
   rocm.bindings.hip.hipBindTexture
   rocm.bindings.hip.hipBindTexture2D
   rocm.bindings.hip.hipBindTextureToArray
   rocm.bindings.hip.hipGetTextureAlignmentOffset
   rocm.bindings.hip.hipUnbindTexture
   rocm.bindings.hip.hipTexRefGetAddress
   rocm.bindings.hip.hipTexRefGetAddressMode
   rocm.bindings.hip.hipTexRefGetFilterMode
   rocm.bindings.hip.hipTexRefGetFlags
   rocm.bindings.hip.hipTexRefGetFormat
   rocm.bindings.hip.hipTexRefGetMaxAnisotropy
   rocm.bindings.hip.hipTexRefGetMipmapFilterMode
   rocm.bindings.hip.hipTexRefGetMipmapLevelBias
   rocm.bindings.hip.hipTexRefGetMipmapLevelClamp
   rocm.bindings.hip.hipTexRefGetMipMappedArray
   rocm.bindings.hip.hipTexRefSetAddress
   rocm.bindings.hip.hipTexRefSetAddress2D
   rocm.bindings.hip.hipTexRefSetMaxAnisotropy
   rocm.bindings.hip.hipTexRefSetBorderColor
   rocm.bindings.hip.hipTexRefSetMipmapFilterMode
   rocm.bindings.hip.hipTexRefSetMipmapLevelBias
   rocm.bindings.hip.hipTexRefSetMipmapLevelClamp
   rocm.bindings.hip.hipTexRefSetMipmappedArray
   rocm.bindings.hip.hipApiName
   rocm.bindings.hip.hipKernelNameRef
   rocm.bindings.hip.hipKernelNameRefByPtr
   rocm.bindings.hip.hipGetStreamDeviceId
   rocm.bindings.hip.hipStreamBeginCapture
   rocm.bindings.hip.hipStreamBeginCaptureToGraph
   rocm.bindings.hip.hipStreamEndCapture
   rocm.bindings.hip.hipStreamGetCaptureInfo
   rocm.bindings.hip.hipStreamGetCaptureInfo_v2
   rocm.bindings.hip.hipStreamIsCapturing
   rocm.bindings.hip.hipStreamUpdateCaptureDependencies
   rocm.bindings.hip.hipThreadExchangeStreamCaptureMode
   rocm.bindings.hip.hipGraphCreate
   rocm.bindings.hip.hipGraphDestroy
   rocm.bindings.hip.hipGraphAddDependencies
   rocm.bindings.hip.hipGraphRemoveDependencies
   rocm.bindings.hip.hipGraphGetEdges
   rocm.bindings.hip.hipGraphGetNodes
   rocm.bindings.hip.hipGraphGetRootNodes
   rocm.bindings.hip.hipGraphNodeGetDependencies
   rocm.bindings.hip.hipGraphNodeGetDependentNodes
   rocm.bindings.hip.hipGraphNodeGetType
   rocm.bindings.hip.hipGraphDestroyNode
   rocm.bindings.hip.hipGraphClone
   rocm.bindings.hip.hipGraphNodeFindInClone
   rocm.bindings.hip.hipGraphInstantiate
   rocm.bindings.hip.hipGraphInstantiateWithFlags
   rocm.bindings.hip.hipGraphInstantiateWithParams
   rocm.bindings.hip.hipGraphLaunch
   rocm.bindings.hip.hipGraphUpload
   rocm.bindings.hip.hipGraphAddNode
   rocm.bindings.hip.hipGraphExecGetFlags
   rocm.bindings.hip.hipGraphNodeSetParams
   rocm.bindings.hip.hipGraphExecNodeSetParams
   rocm.bindings.hip.hipGraphExecDestroy
   rocm.bindings.hip.hipGraphExecUpdate
   rocm.bindings.hip.hipGraphAddKernelNode
   rocm.bindings.hip.hipGraphKernelNodeGetParams
   rocm.bindings.hip.hipGraphKernelNodeSetParams
   rocm.bindings.hip.hipGraphExecKernelNodeSetParams
   rocm.bindings.hip.hipDrvGraphAddMemcpyNode
   rocm.bindings.hip.hipGraphAddMemcpyNode
   rocm.bindings.hip.hipGraphMemcpyNodeGetParams
   rocm.bindings.hip.hipGraphMemcpyNodeSetParams
   rocm.bindings.hip.hipGraphKernelNodeSetAttribute
   rocm.bindings.hip.hipGraphKernelNodeGetAttribute
   rocm.bindings.hip.hipGraphExecMemcpyNodeSetParams
   rocm.bindings.hip.hipGraphAddMemcpyNode1D
   rocm.bindings.hip.hipGraphMemcpyNodeSetParams1D
   rocm.bindings.hip.hipGraphExecMemcpyNodeSetParams1D
   rocm.bindings.hip.hipGraphAddMemcpyNodeFromSymbol
   rocm.bindings.hip.hipGraphMemcpyNodeSetParamsFromSymbol
   rocm.bindings.hip.hipGraphExecMemcpyNodeSetParamsFromSymbol
   rocm.bindings.hip.hipGraphAddMemcpyNodeToSymbol
   rocm.bindings.hip.hipGraphMemcpyNodeSetParamsToSymbol
   rocm.bindings.hip.hipGraphExecMemcpyNodeSetParamsToSymbol
   rocm.bindings.hip.hipGraphAddMemsetNode
   rocm.bindings.hip.hipGraphMemsetNodeGetParams
   rocm.bindings.hip.hipGraphMemsetNodeSetParams
   rocm.bindings.hip.hipGraphExecMemsetNodeSetParams
   rocm.bindings.hip.hipGraphAddHostNode
   rocm.bindings.hip.hipGraphHostNodeGetParams
   rocm.bindings.hip.hipGraphHostNodeSetParams
   rocm.bindings.hip.hipGraphExecHostNodeSetParams
   rocm.bindings.hip.hipGraphAddChildGraphNode
   rocm.bindings.hip.hipGraphChildGraphNodeGetGraph
   rocm.bindings.hip.hipGraphExecChildGraphNodeSetParams
   rocm.bindings.hip.hipGraphAddEmptyNode
   rocm.bindings.hip.hipGraphAddEventRecordNode
   rocm.bindings.hip.hipGraphEventRecordNodeGetEvent
   rocm.bindings.hip.hipGraphEventRecordNodeSetEvent
   rocm.bindings.hip.hipGraphExecEventRecordNodeSetEvent
   rocm.bindings.hip.hipGraphAddEventWaitNode
   rocm.bindings.hip.hipGraphEventWaitNodeGetEvent
   rocm.bindings.hip.hipGraphEventWaitNodeSetEvent
   rocm.bindings.hip.hipGraphExecEventWaitNodeSetEvent
   rocm.bindings.hip.hipGraphAddMemAllocNode
   rocm.bindings.hip.hipGraphMemAllocNodeGetParams
   rocm.bindings.hip.hipGraphAddMemFreeNode
   rocm.bindings.hip.hipGraphMemFreeNodeGetParams
   rocm.bindings.hip.hipDeviceGetGraphMemAttribute
   rocm.bindings.hip.hipDeviceSetGraphMemAttribute
   rocm.bindings.hip.hipDeviceGraphMemTrim
   rocm.bindings.hip.hipUserObjectCreate
   rocm.bindings.hip.hipUserObjectRelease
   rocm.bindings.hip.hipUserObjectRetain
   rocm.bindings.hip.hipGraphRetainUserObject
   rocm.bindings.hip.hipGraphReleaseUserObject
   rocm.bindings.hip.hipGraphDebugDotPrint
   rocm.bindings.hip.hipGraphKernelNodeCopyAttributes
   rocm.bindings.hip.hipGraphNodeSetEnabled
   rocm.bindings.hip.hipGraphNodeGetEnabled
   rocm.bindings.hip.hipGraphAddExternalSemaphoresWaitNode
   rocm.bindings.hip.hipGraphAddExternalSemaphoresSignalNode
   rocm.bindings.hip.hipGraphExternalSemaphoresSignalNodeSetParams
   rocm.bindings.hip.hipGraphExternalSemaphoresWaitNodeSetParams
   rocm.bindings.hip.hipGraphExternalSemaphoresSignalNodeGetParams
   rocm.bindings.hip.hipGraphExternalSemaphoresWaitNodeGetParams
   rocm.bindings.hip.hipGraphExecExternalSemaphoresSignalNodeSetParams
   rocm.bindings.hip.hipGraphExecExternalSemaphoresWaitNodeSetParams
   rocm.bindings.hip.hipDrvGraphMemcpyNodeGetParams
   rocm.bindings.hip.hipDrvGraphMemcpyNodeSetParams
   rocm.bindings.hip.hipDrvGraphAddMemsetNode
   rocm.bindings.hip.hipDrvGraphAddMemFreeNode
   rocm.bindings.hip.hipDrvGraphExecMemcpyNodeSetParams
   rocm.bindings.hip.hipDrvGraphExecMemsetNodeSetParams
   rocm.bindings.hip.hipMemAddressFree
   rocm.bindings.hip.hipMemAddressReserve
   rocm.bindings.hip.hipMemCreate
   rocm.bindings.hip.hipMemExportToShareableHandle
   rocm.bindings.hip.hipMemGetAccess
   rocm.bindings.hip.hipMemGetAllocationGranularity
   rocm.bindings.hip.hipMemGetAllocationPropertiesFromHandle
   rocm.bindings.hip.hipMemImportFromShareableHandle
   rocm.bindings.hip.hipMemMap
   rocm.bindings.hip.hipMemMapArrayAsync
   rocm.bindings.hip.hipMemRelease
   rocm.bindings.hip.hipMemRetainAllocationHandle
   rocm.bindings.hip.hipMemSetAccess
   rocm.bindings.hip.hipMemUnmap
   rocm.bindings.hip.hipGraphicsMapResources
   rocm.bindings.hip.hipGraphicsSubResourceGetMappedArray
   rocm.bindings.hip.hipGraphicsResourceGetMappedPointer
   rocm.bindings.hip.hipGraphicsUnmapResources
   rocm.bindings.hip.hipGraphicsUnregisterResource
   rocm.bindings.hip.hipCreateSurfaceObject
   rocm.bindings.hip.hipDestroySurfaceObject
   rocm.bindings.hip.hipExtEnableLogging
   rocm.bindings.hip.hipExtDisableLogging
   rocm.bindings.hip.hipExtSetLoggingParams
   rocm.bindings.hip.hipMemcpy_spt
   rocm.bindings.hip.hipMemcpyToSymbol_spt
   rocm.bindings.hip.hipMemcpyFromSymbol_spt
   rocm.bindings.hip.hipMemcpy2D_spt
   rocm.bindings.hip.hipMemcpy2DFromArray_spt
   rocm.bindings.hip.hipMemcpy3D_spt
   rocm.bindings.hip.hipMemset_spt
   rocm.bindings.hip.hipMemsetAsync_spt
   rocm.bindings.hip.hipMemset2D_spt
   rocm.bindings.hip.hipMemset2DAsync_spt
   rocm.bindings.hip.hipMemset3DAsync_spt
   rocm.bindings.hip.hipMemset3D_spt
   rocm.bindings.hip.hipMemcpyAsync_spt
   rocm.bindings.hip.hipMemcpy3DAsync_spt
   rocm.bindings.hip.hipMemcpy2DAsync_spt
   rocm.bindings.hip.hipMemcpyFromSymbolAsync_spt
   rocm.bindings.hip.hipMemcpyToSymbolAsync_spt
   rocm.bindings.hip.hipMemcpyFromArray_spt
   rocm.bindings.hip.hipMemcpy2DToArray_spt
   rocm.bindings.hip.hipMemcpy2DFromArrayAsync_spt
   rocm.bindings.hip.hipMemcpy2DToArrayAsync_spt
   rocm.bindings.hip.hipStreamQuery_spt
   rocm.bindings.hip.hipStreamSynchronize_spt
   rocm.bindings.hip.hipStreamGetPriority_spt
   rocm.bindings.hip.hipStreamWaitEvent_spt
   rocm.bindings.hip.hipStreamGetFlags_spt
   rocm.bindings.hip.hipStreamAddCallback_spt
   rocm.bindings.hip.hipEventRecord_spt
   rocm.bindings.hip.hipLaunchCooperativeKernel_spt
   rocm.bindings.hip.hipLaunchKernel_spt
   rocm.bindings.hip.hipGraphLaunch_spt
   rocm.bindings.hip.hipStreamBeginCapture_spt
   rocm.bindings.hip.hipStreamEndCapture_spt
   rocm.bindings.hip.hipStreamIsCapturing_spt
   rocm.bindings.hip.hipStreamGetCaptureInfo_spt
   rocm.bindings.hip.hipStreamGetCaptureInfo_v2_spt
   rocm.bindings.hip.hipLaunchHostFunc_spt
   rocm.bindings.hip.hipGetDriverEntryPoint_spt
   rocm.bindings.hip.hipGetProcAddress_spt


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: HIP_VERSION_MAJOR
   :type:  Any

.. py:data:: HIP_VERSION_MINOR
   :type:  Any

.. py:data:: HIP_VERSION_PATCH
   :type:  Any

.. py:data:: HIP_VERSION_GITHASH
   :type:  Any

.. py:data:: HIP_VERSION_BUILD_ID
   :type:  Any

.. py:data:: HIP_VERSION_BUILD_NAME
   :type:  Any

.. py:data:: HIP_VERSION
   :type:  Any

.. py:data:: HIP_TRSA_OVERRIDE_FORMAT
   :type:  Any

.. py:data:: HIP_TRSF_READ_AS_INTEGER
   :type:  Any

.. py:data:: HIP_TRSF_NORMALIZED_COORDINATES
   :type:  Any

.. py:data:: HIP_TRSF_SRGB
   :type:  Any

.. py:data:: hipTextureType1D
   :type:  Any

.. py:data:: hipTextureType2D
   :type:  Any

.. py:data:: hipTextureType3D
   :type:  Any

.. py:data:: hipTextureTypeCubemap
   :type:  Any

.. py:data:: hipTextureType1DLayered
   :type:  Any

.. py:data:: hipTextureType2DLayered
   :type:  Any

.. py:data:: hipTextureTypeCubemapLayered
   :type:  Any

.. py:data:: HIP_IMAGE_OBJECT_SIZE_DWORD
   :type:  Any

.. py:data:: HIP_SAMPLER_OBJECT_SIZE_DWORD
   :type:  Any

.. py:data:: HIP_SAMPLER_OBJECT_OFFSET_DWORD
   :type:  Any

.. py:data:: HIP_TEXTURE_OBJECT_SIZE_DWORD
   :type:  Any

.. py:data:: HIP_LAUNCH_PARAM_BUFFER_POINTER
   :type:  Any

.. py:data:: HIP_LAUNCH_PARAM_BUFFER_SIZE
   :type:  Any

.. py:data:: HIP_LAUNCH_PARAM_END
   :type:  Any

.. py:data:: hipIpcMemLazyEnablePeerAccess
   :type:  Any

.. py:data:: HIP_IPC_HANDLE_SIZE
   :type:  Any

.. py:data:: hipStreamDefault
   :type:  Any

.. py:data:: hipStreamNonBlocking
   :type:  Any

.. py:data:: hipEventDefault
   :type:  Any

.. py:data:: hipEventBlockingSync
   :type:  Any

.. py:data:: hipEventDisableTiming
   :type:  Any

.. py:data:: hipEventInterprocess
   :type:  Any

.. py:data:: hipEventReleaseToDevice
   :type:  Any

.. py:data:: hipEventReleaseToSystem
   :type:  Any

.. py:data:: hipHostMallocDefault
   :type:  Any

.. py:data:: hipHostMallocPortable
   :type:  Any

.. py:data:: hipHostMallocMapped
   :type:  Any

.. py:data:: hipHostMallocWriteCombined
   :type:  Any

.. py:data:: hipHostMallocNumaUser
   :type:  Any

.. py:data:: hipHostMallocCoherent
   :type:  Any

.. py:data:: hipHostMallocNonCoherent
   :type:  Any

.. py:data:: hipMemAttachGlobal
   :type:  Any

.. py:data:: hipMemAttachHost
   :type:  Any

.. py:data:: hipMemAttachSingle
   :type:  Any

.. py:data:: hipDeviceMallocDefault
   :type:  Any

.. py:data:: hipDeviceMallocFinegrained
   :type:  Any

.. py:data:: hipMallocSignalMemory
   :type:  Any

.. py:data:: hipHostRegisterDefault
   :type:  Any

.. py:data:: hipHostRegisterPortable
   :type:  Any

.. py:data:: hipHostRegisterMapped
   :type:  Any

.. py:data:: hipHostRegisterIoMemory
   :type:  Any

.. py:data:: hipExtHostRegisterCoarseGrained
   :type:  Any

.. py:data:: hipDeviceScheduleAuto
   :type:  Any

.. py:data:: hipDeviceScheduleSpin
   :type:  Any

.. py:data:: hipDeviceScheduleYield
   :type:  Any

.. py:data:: hipDeviceScheduleBlockingSync
   :type:  Any

.. py:data:: hipDeviceScheduleMask
   :type:  Any

.. py:data:: hipDeviceMapHost
   :type:  Any

.. py:data:: hipDeviceLmemResizeToMax
   :type:  Any

.. py:data:: hipArrayDefault
   :type:  Any

.. py:data:: hipArrayLayered
   :type:  Any

.. py:data:: hipArraySurfaceLoadStore
   :type:  Any

.. py:data:: hipArrayCubemap
   :type:  Any

.. py:data:: hipArrayTextureGather
   :type:  Any

.. py:data:: hipOccupancyDefault
   :type:  Any

.. py:data:: hipCooperativeLaunchMultiDeviceNoPreSync
   :type:  Any

.. py:data:: hipCooperativeLaunchMultiDeviceNoPostSync
   :type:  Any

.. py:data:: hipCpuDeviceId
   :type:  Any

.. py:data:: hipInvalidDeviceId
   :type:  Any

.. py:data:: hipExtAnyOrderLaunch
   :type:  Any

.. py:data:: hipStreamWaitValueGte
   :type:  Any

.. py:data:: hipStreamWaitValueEq
   :type:  Any

.. py:data:: hipStreamWaitValueAnd
   :type:  Any

.. py:data:: hipStreamWaitValueNor
   :type:  Any

.. py:class:: hipJitOption

   Bases: :py:obj:`enum.IntEnum`


   hipJitOption
       


   .. py:attribute:: hipJitOptionMaxRegisters
      :type:  int


   .. py:attribute:: hipJitOptionThreadsPerBlock
      :type:  int


   .. py:attribute:: hipJitOptionWallTime
      :type:  int


   .. py:attribute:: hipJitOptionInfoLogBuffer
      :type:  int


   .. py:attribute:: hipJitOptionInfoLogBufferSizeBytes
      :type:  int


   .. py:attribute:: hipJitOptionErrorLogBuffer
      :type:  int


   .. py:attribute:: hipJitOptionErrorLogBufferSizeBytes
      :type:  int


   .. py:attribute:: hipJitOptionOptimizationLevel
      :type:  int


   .. py:attribute:: hipJitOptionTargetFromContext
      :type:  int


   .. py:attribute:: hipJitOptionTarget
      :type:  int


   .. py:attribute:: hipJitOptionFallbackStrategy
      :type:  int


   .. py:attribute:: hipJitOptionGenerateDebugInfo
      :type:  int


   .. py:attribute:: hipJitOptionLogVerbose
      :type:  int


   .. py:attribute:: hipJitOptionGenerateLineInfo
      :type:  int


   .. py:attribute:: hipJitOptionCacheMode
      :type:  int


   .. py:attribute:: hipJitOptionSm3xOpt
      :type:  int


   .. py:attribute:: hipJitOptionFastCompile
      :type:  int


   .. py:attribute:: hipJitOptionGlobalSymbolNames
      :type:  int


   .. py:attribute:: hipJitOptionGlobalSymbolAddresses
      :type:  int


   .. py:attribute:: hipJitOptionGlobalSymbolCount
      :type:  int


   .. py:attribute:: hipJitOptionLto
      :type:  int


   .. py:attribute:: hipJitOptionFtz
      :type:  int


   .. py:attribute:: hipJitOptionPrecDiv
      :type:  int


   .. py:attribute:: hipJitOptionPrecSqrt
      :type:  int


   .. py:attribute:: hipJitOptionFma
      :type:  int


   .. py:attribute:: hipJitOptionPositionIndependentCode
      :type:  int


   .. py:attribute:: hipJitOptionMinCTAPerSM
      :type:  int


   .. py:attribute:: hipJitOptionMaxThreadsPerBlock
      :type:  int


   .. py:attribute:: hipJitOptionOverrideDirectiveValues
      :type:  int


   .. py:attribute:: hipJitOptionNumOptions
      :type:  int


   .. py:attribute:: hipJitOptionIRtoISAOptExt
      :type:  int


   .. py:attribute:: hipJitOptionIRtoISAOptCountExt
      :type:  int


.. py:class:: hipJitInputType

   Bases: :py:obj:`enum.IntEnum`


   hipJitInputType
       


   .. py:attribute:: hipJitInputCubin
      :type:  int


   .. py:attribute:: hipJitInputPtx
      :type:  int


   .. py:attribute:: hipJitInputFatBinary
      :type:  int


   .. py:attribute:: hipJitInputObject
      :type:  int


   .. py:attribute:: hipJitInputLibrary
      :type:  int


   .. py:attribute:: hipJitInputNvvm
      :type:  int


   .. py:attribute:: hipJitNumLegacyInputTypes
      :type:  int


   .. py:attribute:: hipJitInputLLVMBitcode
      :type:  int


   .. py:attribute:: hipJitInputLLVMBundledBitcode
      :type:  int


   .. py:attribute:: hipJitInputLLVMArchivesOfBundledBitcode
      :type:  int


   .. py:attribute:: hipJitInputSpirv
      :type:  int


   .. py:attribute:: hipJitNumInputTypes
      :type:  int


.. py:class:: hipJitCacheMode

   Bases: :py:obj:`enum.IntEnum`


   hipJitCacheMode
       


   .. py:attribute:: hipJitCacheOptionNone
      :type:  int


   .. py:attribute:: hipJitCacheOptionCG
      :type:  int


   .. py:attribute:: hipJitCacheOptionCA
      :type:  int


.. py:class:: hipJitFallback

   Bases: :py:obj:`enum.IntEnum`


   hipJitFallback
       


   .. py:attribute:: hipJitPreferPTX
      :type:  int


   .. py:attribute:: hipJitPreferBinary
      :type:  int


.. py:class:: hipLibraryOption_e

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipLibraryHostUniversalFunctionAndDataTable
      :type:  int


   .. py:attribute:: hipLibraryBinaryIsPreserved
      :type:  int


.. py:data:: hipLibraryOption

.. py:data:: HIP_SUCCESS
   :type:  int

.. py:data:: HIP_ERROR_INVALID_VALUE
   :type:  int

.. py:data:: HIP_ERROR_NOT_INITIALIZED
   :type:  int

.. py:data:: HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
   :type:  int

.. py:class:: hipDeviceArch_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   *

   hipDeviceArch_t


   .. py:attribute:: hasGlobalInt32Atomics
      :type:  Any


   .. py:attribute:: hasGlobalFloatAtomicExch
      :type:  Any


   .. py:attribute:: hasSharedInt32Atomics
      :type:  Any


   .. py:attribute:: hasSharedFloatAtomicExch
      :type:  Any


   .. py:attribute:: hasFloatAtomicAdd
      :type:  Any


   .. py:attribute:: hasGlobalInt64Atomics
      :type:  Any


   .. py:attribute:: hasSharedInt64Atomics
      :type:  Any


   .. py:attribute:: hasDoubles
      :type:  Any


   .. py:attribute:: hasWarpVote
      :type:  Any


   .. py:attribute:: hasWarpBallot
      :type:  Any


   .. py:attribute:: hasWarpShuffle
      :type:  Any


   .. py:attribute:: hasFunnelShift
      :type:  Any


   .. py:attribute:: hasThreadFenceSystem
      :type:  Any


   .. py:attribute:: hasSyncThreadsExt
      :type:  Any


   .. py:attribute:: hasSurfaceFuncs
      :type:  Any


   .. py:attribute:: has3dGrid
      :type:  Any


   .. py:attribute:: hasDynamicParallelism
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipUUID_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: bytes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipUUID

.. py:class:: hipDeviceProp_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   hipDeviceProp
       


   .. py:attribute:: name
      :type:  Any


   .. py:attribute:: uuid
      :type:  Any


   .. py:attribute:: luid
      :type:  Any


   .. py:attribute:: luidDeviceNodeMask
      :type:  Any


   .. py:attribute:: totalGlobalMem
      :type:  Any


   .. py:attribute:: sharedMemPerBlock
      :type:  Any


   .. py:attribute:: regsPerBlock
      :type:  Any


   .. py:attribute:: warpSize
      :type:  Any


   .. py:attribute:: memPitch
      :type:  Any


   .. py:attribute:: maxThreadsPerBlock
      :type:  Any


   .. py:attribute:: maxThreadsDim
      :type:  Any


   .. py:attribute:: maxGridSize
      :type:  Any


   .. py:attribute:: clockRate
      :type:  Any


   .. py:attribute:: totalConstMem
      :type:  Any


   .. py:attribute:: major
      :type:  Any


   .. py:attribute:: minor
      :type:  Any


   .. py:attribute:: textureAlignment
      :type:  Any


   .. py:attribute:: texturePitchAlignment
      :type:  Any


   .. py:attribute:: deviceOverlap
      :type:  Any


   .. py:attribute:: multiProcessorCount
      :type:  Any


   .. py:attribute:: kernelExecTimeoutEnabled
      :type:  Any


   .. py:attribute:: integrated
      :type:  Any


   .. py:attribute:: canMapHostMemory
      :type:  Any


   .. py:attribute:: computeMode
      :type:  Any


   .. py:attribute:: maxTexture1D
      :type:  Any


   .. py:attribute:: maxTexture1DMipmap
      :type:  Any


   .. py:attribute:: maxTexture1DLinear
      :type:  Any


   .. py:attribute:: maxTexture2D
      :type:  Any


   .. py:attribute:: maxTexture2DMipmap
      :type:  Any


   .. py:attribute:: maxTexture2DLinear
      :type:  Any


   .. py:attribute:: maxTexture2DGather
      :type:  Any


   .. py:attribute:: maxTexture3D
      :type:  Any


   .. py:attribute:: maxTexture3DAlt
      :type:  Any


   .. py:attribute:: maxTextureCubemap
      :type:  Any


   .. py:attribute:: maxTexture1DLayered
      :type:  Any


   .. py:attribute:: maxTexture2DLayered
      :type:  Any


   .. py:attribute:: maxTextureCubemapLayered
      :type:  Any


   .. py:attribute:: maxSurface1D
      :type:  Any


   .. py:attribute:: maxSurface2D
      :type:  Any


   .. py:attribute:: maxSurface3D
      :type:  Any


   .. py:attribute:: maxSurface1DLayered
      :type:  Any


   .. py:attribute:: maxSurface2DLayered
      :type:  Any


   .. py:attribute:: maxSurfaceCubemap
      :type:  Any


   .. py:attribute:: maxSurfaceCubemapLayered
      :type:  Any


   .. py:attribute:: surfaceAlignment
      :type:  Any


   .. py:attribute:: concurrentKernels
      :type:  Any


   .. py:attribute:: ECCEnabled
      :type:  Any


   .. py:attribute:: pciBusID
      :type:  Any


   .. py:attribute:: pciDeviceID
      :type:  Any


   .. py:attribute:: pciDomainID
      :type:  Any


   .. py:attribute:: tccDriver
      :type:  Any


   .. py:attribute:: asyncEngineCount
      :type:  Any


   .. py:attribute:: unifiedAddressing
      :type:  Any


   .. py:attribute:: memoryClockRate
      :type:  Any


   .. py:attribute:: memoryBusWidth
      :type:  Any


   .. py:attribute:: l2CacheSize
      :type:  Any


   .. py:attribute:: persistingL2CacheMaxSize
      :type:  Any


   .. py:attribute:: maxThreadsPerMultiProcessor
      :type:  Any


   .. py:attribute:: streamPrioritiesSupported
      :type:  Any


   .. py:attribute:: globalL1CacheSupported
      :type:  Any


   .. py:attribute:: localL1CacheSupported
      :type:  Any


   .. py:attribute:: sharedMemPerMultiprocessor
      :type:  Any


   .. py:attribute:: regsPerMultiprocessor
      :type:  Any


   .. py:attribute:: managedMemory
      :type:  Any


   .. py:attribute:: isMultiGpuBoard
      :type:  Any


   .. py:attribute:: multiGpuBoardGroupID
      :type:  Any


   .. py:attribute:: hostNativeAtomicSupported
      :type:  Any


   .. py:attribute:: singleToDoublePrecisionPerfRatio
      :type:  Any


   .. py:attribute:: pageableMemoryAccess
      :type:  Any


   .. py:attribute:: concurrentManagedAccess
      :type:  Any


   .. py:attribute:: computePreemptionSupported
      :type:  Any


   .. py:attribute:: canUseHostPointerForRegisteredMem
      :type:  Any


   .. py:attribute:: cooperativeLaunch
      :type:  Any


   .. py:attribute:: cooperativeMultiDeviceLaunch
      :type:  Any


   .. py:attribute:: sharedMemPerBlockOptin
      :type:  Any


   .. py:attribute:: pageableMemoryAccessUsesHostPageTables
      :type:  Any


   .. py:attribute:: directManagedMemAccessFromHost
      :type:  Any


   .. py:attribute:: maxBlocksPerMultiProcessor
      :type:  Any


   .. py:attribute:: accessPolicyMaxWindowSize
      :type:  Any


   .. py:attribute:: reservedSharedMemPerBlock
      :type:  Any


   .. py:attribute:: hostRegisterSupported
      :type:  Any


   .. py:attribute:: sparseHipArraySupported
      :type:  Any


   .. py:attribute:: hostRegisterReadOnlySupported
      :type:  Any


   .. py:attribute:: timelineSemaphoreInteropSupported
      :type:  Any


   .. py:attribute:: memoryPoolsSupported
      :type:  Any


   .. py:attribute:: gpuDirectRDMASupported
      :type:  Any


   .. py:attribute:: gpuDirectRDMAFlushWritesOptions
      :type:  Any


   .. py:attribute:: gpuDirectRDMAWritesOrdering
      :type:  Any


   .. py:attribute:: memoryPoolSupportedHandleTypes
      :type:  Any


   .. py:attribute:: deferredMappingHipArraySupported
      :type:  Any


   .. py:attribute:: ipcEventSupported
      :type:  Any


   .. py:attribute:: clusterLaunch
      :type:  Any


   .. py:attribute:: unifiedFunctionPointers
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:attribute:: hipReserved
      :type:  Any


   .. py:attribute:: gcnArchName
      :type:  Any


   .. py:attribute:: maxSharedMemoryPerMultiProcessor
      :type:  Any


   .. py:attribute:: clockInstructionRate
      :type:  Any


   .. py:attribute:: arch
      :type:  Any


   .. py:attribute:: hdpMemFlushCntl
      :type:  Any


   .. py:attribute:: hdpRegFlushCntl
      :type:  Any


   .. py:attribute:: cooperativeMultiDeviceUnmatchedFunc
      :type:  Any


   .. py:attribute:: cooperativeMultiDeviceUnmatchedGridDim
      :type:  Any


   .. py:attribute:: cooperativeMultiDeviceUnmatchedBlockDim
      :type:  Any


   .. py:attribute:: cooperativeMultiDeviceUnmatchedSharedMem
      :type:  Any


   .. py:attribute:: isLargeBar
      :type:  Any


   .. py:attribute:: asicRevision
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemoryType

   Bases: :py:obj:`enum.IntEnum`


   hipMemoryType (for pointer attributes)

   Note:
       hipMemoryType enum values are combination of cudaMemoryType and cuMemoryType and AMD
       specific enum values.


   .. py:attribute:: hipMemoryTypeUnregistered
      :type:  int


   .. py:attribute:: hipMemoryTypeHost
      :type:  int


   .. py:attribute:: hipMemoryTypeDevice
      :type:  int


   .. py:attribute:: hipMemoryTypeManaged
      :type:  int


   .. py:attribute:: hipMemoryTypeArray
      :type:  int


   .. py:attribute:: hipMemoryTypeUnified
      :type:  int


.. py:class:: hipPointerAttribute_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Pointer attributes
       


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: device
      :type:  Any


   .. py:attribute:: devicePointer
      :type:  Any


   .. py:attribute:: hostPointer
      :type:  Any


   .. py:attribute:: isManaged
      :type:  Any


   .. py:attribute:: allocationFlags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipError_t

   Bases: :py:obj:`enum.IntEnum`


   HIP error type
       


   .. py:attribute:: hipSuccess
      :type:  int


   .. py:attribute:: hipErrorInvalidValue
      :type:  int


   .. py:attribute:: hipErrorOutOfMemory
      :type:  int


   .. py:attribute:: hipErrorMemoryAllocation
      :type:  int


   .. py:attribute:: hipErrorNotInitialized
      :type:  int


   .. py:attribute:: hipErrorInitializationError
      :type:  int


   .. py:attribute:: hipErrorDeinitialized
      :type:  int


   .. py:attribute:: hipErrorProfilerDisabled
      :type:  int


   .. py:attribute:: hipErrorProfilerNotInitialized
      :type:  int


   .. py:attribute:: hipErrorProfilerAlreadyStarted
      :type:  int


   .. py:attribute:: hipErrorProfilerAlreadyStopped
      :type:  int


   .. py:attribute:: hipErrorInvalidConfiguration
      :type:  int


   .. py:attribute:: hipErrorInvalidPitchValue
      :type:  int


   .. py:attribute:: hipErrorInvalidSymbol
      :type:  int


   .. py:attribute:: hipErrorInvalidDevicePointer
      :type:  int


   .. py:attribute:: hipErrorInvalidMemcpyDirection
      :type:  int


   .. py:attribute:: hipErrorInsufficientDriver
      :type:  int


   .. py:attribute:: hipErrorMissingConfiguration
      :type:  int


   .. py:attribute:: hipErrorPriorLaunchFailure
      :type:  int


   .. py:attribute:: hipErrorInvalidDeviceFunction
      :type:  int


   .. py:attribute:: hipErrorNoDevice
      :type:  int


   .. py:attribute:: hipErrorInvalidDevice
      :type:  int


   .. py:attribute:: hipErrorInvalidImage
      :type:  int


   .. py:attribute:: hipErrorInvalidContext
      :type:  int


   .. py:attribute:: hipErrorContextAlreadyCurrent
      :type:  int


   .. py:attribute:: hipErrorMapFailed
      :type:  int


   .. py:attribute:: hipErrorMapBufferObjectFailed
      :type:  int


   .. py:attribute:: hipErrorUnmapFailed
      :type:  int


   .. py:attribute:: hipErrorArrayIsMapped
      :type:  int


   .. py:attribute:: hipErrorAlreadyMapped
      :type:  int


   .. py:attribute:: hipErrorNoBinaryForGpu
      :type:  int


   .. py:attribute:: hipErrorAlreadyAcquired
      :type:  int


   .. py:attribute:: hipErrorNotMapped
      :type:  int


   .. py:attribute:: hipErrorNotMappedAsArray
      :type:  int


   .. py:attribute:: hipErrorNotMappedAsPointer
      :type:  int


   .. py:attribute:: hipErrorECCNotCorrectable
      :type:  int


   .. py:attribute:: hipErrorUnsupportedLimit
      :type:  int


   .. py:attribute:: hipErrorContextAlreadyInUse
      :type:  int


   .. py:attribute:: hipErrorPeerAccessUnsupported
      :type:  int


   .. py:attribute:: hipErrorInvalidKernelFile
      :type:  int


   .. py:attribute:: hipErrorInvalidGraphicsContext
      :type:  int


   .. py:attribute:: hipErrorInvalidSource
      :type:  int


   .. py:attribute:: hipErrorFileNotFound
      :type:  int


   .. py:attribute:: hipErrorSharedObjectSymbolNotFound
      :type:  int


   .. py:attribute:: hipErrorSharedObjectInitFailed
      :type:  int


   .. py:attribute:: hipErrorOperatingSystem
      :type:  int


   .. py:attribute:: hipErrorInvalidHandle
      :type:  int


   .. py:attribute:: hipErrorInvalidResourceHandle
      :type:  int


   .. py:attribute:: hipErrorIllegalState
      :type:  int


   .. py:attribute:: hipErrorNotFound
      :type:  int


   .. py:attribute:: hipErrorNotReady
      :type:  int


   .. py:attribute:: hipErrorIllegalAddress
      :type:  int


   .. py:attribute:: hipErrorLaunchOutOfResources
      :type:  int


   .. py:attribute:: hipErrorLaunchTimeOut
      :type:  int


   .. py:attribute:: hipErrorPeerAccessAlreadyEnabled
      :type:  int


   .. py:attribute:: hipErrorPeerAccessNotEnabled
      :type:  int


   .. py:attribute:: hipErrorSetOnActiveProcess
      :type:  int


   .. py:attribute:: hipErrorContextIsDestroyed
      :type:  int


   .. py:attribute:: hipErrorAssert
      :type:  int


   .. py:attribute:: hipErrorHostMemoryAlreadyRegistered
      :type:  int


   .. py:attribute:: hipErrorHostMemoryNotRegistered
      :type:  int


   .. py:attribute:: hipErrorLaunchFailure
      :type:  int


   .. py:attribute:: hipErrorCooperativeLaunchTooLarge
      :type:  int


   .. py:attribute:: hipErrorNotSupported
      :type:  int


   .. py:attribute:: hipErrorStreamCaptureUnsupported
      :type:  int


   .. py:attribute:: hipErrorStreamCaptureInvalidated
      :type:  int


   .. py:attribute:: hipErrorStreamCaptureMerge
      :type:  int


   .. py:attribute:: hipErrorStreamCaptureUnmatched
      :type:  int


   .. py:attribute:: hipErrorStreamCaptureUnjoined
      :type:  int


   .. py:attribute:: hipErrorStreamCaptureIsolation
      :type:  int


   .. py:attribute:: hipErrorStreamCaptureImplicit
      :type:  int


   .. py:attribute:: hipErrorCapturedEvent
      :type:  int


   .. py:attribute:: hipErrorStreamCaptureWrongThread
      :type:  int


   .. py:attribute:: hipErrorGraphExecUpdateFailure
      :type:  int


   .. py:attribute:: hipErrorInvalidChannelDescriptor
      :type:  int


   .. py:attribute:: hipErrorInvalidTexture
      :type:  int


   .. py:attribute:: hipErrorInvalidResourceType
      :type:  int


   .. py:attribute:: hipErrorInvalidResourceConfiguration
      :type:  int


   .. py:attribute:: hipErrorStreamDetached
      :type:  int


   .. py:attribute:: hipErrorUnknown
      :type:  int


   .. py:attribute:: hipErrorRuntimeMemory
      :type:  int


   .. py:attribute:: hipErrorRuntimeOther
      :type:  int


   .. py:attribute:: hipErrorInvalidClusterSize
      :type:  int


   .. py:attribute:: hipErrorTbd
      :type:  int


.. py:class:: hipDeviceAttribute_t

   Bases: :py:obj:`enum.IntEnum`


   hipDeviceAttribute_t hipDeviceAttributeUnused number: 5

   hipDeviceAttribute_t
   hipDeviceAttributeUnused number: 5


   .. py:attribute:: hipDeviceAttributeCudaCompatibleBegin
      :type:  int


   .. py:attribute:: hipDeviceAttributeEccEnabled
      :type:  int


   .. py:attribute:: hipDeviceAttributeAccessPolicyMaxWindowSize
      :type:  int


   .. py:attribute:: hipDeviceAttributeAsyncEngineCount
      :type:  int


   .. py:attribute:: hipDeviceAttributeCanMapHostMemory
      :type:  int


   .. py:attribute:: hipDeviceAttributeCanUseHostPointerForRegisteredMem
      :type:  int


   .. py:attribute:: hipDeviceAttributeClockRate
      :type:  int


   .. py:attribute:: hipDeviceAttributeComputeMode
      :type:  int


   .. py:attribute:: hipDeviceAttributeComputePreemptionSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeConcurrentKernels
      :type:  int


   .. py:attribute:: hipDeviceAttributeConcurrentManagedAccess
      :type:  int


   .. py:attribute:: hipDeviceAttributeCooperativeLaunch
      :type:  int


   .. py:attribute:: hipDeviceAttributeCooperativeMultiDeviceLaunch
      :type:  int


   .. py:attribute:: hipDeviceAttributeDeviceOverlap
      :type:  int


   .. py:attribute:: hipDeviceAttributeDirectManagedMemAccessFromHost
      :type:  int


   .. py:attribute:: hipDeviceAttributeGlobalL1CacheSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeHostNativeAtomicSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeIntegrated
      :type:  int


   .. py:attribute:: hipDeviceAttributeIsMultiGpuBoard
      :type:  int


   .. py:attribute:: hipDeviceAttributeKernelExecTimeout
      :type:  int


   .. py:attribute:: hipDeviceAttributeL2CacheSize
      :type:  int


   .. py:attribute:: hipDeviceAttributeLocalL1CacheSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeLuid
      :type:  int


   .. py:attribute:: hipDeviceAttributeLuidDeviceNodeMask
      :type:  int


   .. py:attribute:: hipDeviceAttributeComputeCapabilityMajor
      :type:  int


   .. py:attribute:: hipDeviceAttributeManagedMemory
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxBlocksPerMultiProcessor
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxBlockDimX
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxBlockDimY
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxBlockDimZ
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxGridDimX
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxGridDimY
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxGridDimZ
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxSurface1D
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxSurface1DLayered
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxSurface2D
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxSurface2DLayered
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxSurface3D
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxSurfaceCubemap
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxSurfaceCubemapLayered
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture1DWidth
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture1DLayered
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture1DLinear
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture1DMipmap
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture2DWidth
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture2DHeight
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture2DGather
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture2DLayered
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture2DLinear
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture2DMipmap
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture3DWidth
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture3DHeight
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture3DDepth
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTexture3DAlt
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTextureCubemap
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxTextureCubemapLayered
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxThreadsDim
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxThreadsPerBlock
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxThreadsPerMultiProcessor
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxPitch
      :type:  int


   .. py:attribute:: hipDeviceAttributeMemoryBusWidth
      :type:  int


   .. py:attribute:: hipDeviceAttributeMemoryClockRate
      :type:  int


   .. py:attribute:: hipDeviceAttributeComputeCapabilityMinor
      :type:  int


   .. py:attribute:: hipDeviceAttributeMultiGpuBoardGroupID
      :type:  int


   .. py:attribute:: hipDeviceAttributeMultiprocessorCount
      :type:  int


   .. py:attribute:: hipDeviceAttributeUnused1
      :type:  int


   .. py:attribute:: hipDeviceAttributePageableMemoryAccess
      :type:  int


   .. py:attribute:: hipDeviceAttributePageableMemoryAccessUsesHostPageTables
      :type:  int


   .. py:attribute:: hipDeviceAttributePciBusId
      :type:  int


   .. py:attribute:: hipDeviceAttributePciDeviceId
      :type:  int


   .. py:attribute:: hipDeviceAttributePciDomainId
      :type:  int


   .. py:attribute:: hipDeviceAttributePciDomainID
      :type:  int


   .. py:attribute:: hipDeviceAttributePersistingL2CacheMaxSize
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxRegistersPerBlock
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxRegistersPerMultiprocessor
      :type:  int


   .. py:attribute:: hipDeviceAttributeReservedSharedMemPerBlock
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxSharedMemoryPerBlock
      :type:  int


   .. py:attribute:: hipDeviceAttributeSharedMemPerBlockOptin
      :type:  int


   .. py:attribute:: hipDeviceAttributeSharedMemPerMultiprocessor
      :type:  int


   .. py:attribute:: hipDeviceAttributeSingleToDoublePrecisionPerfRatio
      :type:  int


   .. py:attribute:: hipDeviceAttributeStreamPrioritiesSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeSurfaceAlignment
      :type:  int


   .. py:attribute:: hipDeviceAttributeTccDriver
      :type:  int


   .. py:attribute:: hipDeviceAttributeTextureAlignment
      :type:  int


   .. py:attribute:: hipDeviceAttributeTexturePitchAlignment
      :type:  int


   .. py:attribute:: hipDeviceAttributeTotalConstantMemory
      :type:  int


   .. py:attribute:: hipDeviceAttributeTotalGlobalMem
      :type:  int


   .. py:attribute:: hipDeviceAttributeUnifiedAddressing
      :type:  int


   .. py:attribute:: hipDeviceAttributeUnused2
      :type:  int


   .. py:attribute:: hipDeviceAttributeWarpSize
      :type:  int


   .. py:attribute:: hipDeviceAttributeMemoryPoolsSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeVirtualMemoryManagementSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeHostRegisterSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeMemoryPoolSupportedHandleTypes
      :type:  int


   .. py:attribute:: hipDeviceAttributeHostNumaId
      :type:  int


   .. py:attribute:: hipDeviceAttributeDmaBufSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeGPUDirectRDMAWithHipVMMSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeHandleTypeFabricSupported
      :type:  int


   .. py:attribute:: hipDeviceAttributeCudaCompatibleEnd
      :type:  int


   .. py:attribute:: hipDeviceAttributeAmdSpecificBegin
      :type:  int


   .. py:attribute:: hipDeviceAttributeClockInstructionRate
      :type:  int


   .. py:attribute:: hipDeviceAttributeUnused3
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
      :type:  int


   .. py:attribute:: hipDeviceAttributeUnused4
      :type:  int


   .. py:attribute:: hipDeviceAttributeUnused5
      :type:  int


   .. py:attribute:: hipDeviceAttributeHdpMemFlushCntl
      :type:  int


   .. py:attribute:: hipDeviceAttributeHdpRegFlushCntl
      :type:  int


   .. py:attribute:: hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
      :type:  int


   .. py:attribute:: hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
      :type:  int


   .. py:attribute:: hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
      :type:  int


   .. py:attribute:: hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem
      :type:  int


   .. py:attribute:: hipDeviceAttributeIsLargeBar
      :type:  int


   .. py:attribute:: hipDeviceAttributeAsicRevision
      :type:  int


   .. py:attribute:: hipDeviceAttributeCanUseStreamWaitValue
      :type:  int


   .. py:attribute:: hipDeviceAttributeImageSupport
      :type:  int


   .. py:attribute:: hipDeviceAttributePhysicalMultiProcessorCount
      :type:  int


   .. py:attribute:: hipDeviceAttributeFineGrainSupport
      :type:  int


   .. py:attribute:: hipDeviceAttributeWallClockRate
      :type:  int


   .. py:attribute:: hipDeviceAttributeNumberOfXccs
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxAvailableVgprsPerThread
      :type:  int


   .. py:attribute:: hipDeviceAttributePciChipId
      :type:  int


   .. py:attribute:: hipDeviceAttributeExpertSchedMode
      :type:  int


   .. py:attribute:: hipDeviceAttributeMaxDynDataPrefetchRegions
      :type:  int


   .. py:attribute:: hipDeviceAttributeAmdSpecificEnd
      :type:  int


   .. py:attribute:: hipDeviceAttributeVendorSpecificBegin
      :type:  int


.. py:class:: hipDriverProcAddressQueryResult

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIP_GET_PROC_ADDRESS_SUCCESS
      :type:  int


   .. py:attribute:: HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
      :type:  int


   .. py:attribute:: HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT
      :type:  int


.. py:class:: hipComputeMode

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipComputeModeDefault
      :type:  int


   .. py:attribute:: hipComputeModeExclusive
      :type:  int


   .. py:attribute:: hipComputeModeProhibited
      :type:  int


   .. py:attribute:: hipComputeModeExclusiveProcess
      :type:  int


.. py:class:: hipFlushGPUDirectRDMAWritesOptions

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipFlushGPUDirectRDMAWritesOptionHost
      :type:  int


   .. py:attribute:: hipFlushGPUDirectRDMAWritesOptionMemOps
      :type:  int


.. py:class:: hipGPUDirectRDMAWritesOrdering

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipGPUDirectRDMAWritesOrderingNone
      :type:  int


   .. py:attribute:: hipGPUDirectRDMAWritesOrderingOwner
      :type:  int


   .. py:attribute:: hipGPUDirectRDMAWritesOrderingAllDevices
      :type:  int


.. py:class:: hipChannelFormatKind

   Bases: :py:obj:`enum.IntEnum`


   HIP channel format kinds
       


   .. py:attribute:: hipChannelFormatKindSigned
      :type:  int


   .. py:attribute:: hipChannelFormatKindUnsigned
      :type:  int


   .. py:attribute:: hipChannelFormatKindFloat
      :type:  int


   .. py:attribute:: hipChannelFormatKindNone
      :type:  int


.. py:class:: hipChannelFormatDesc(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP channel format descriptor
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:attribute:: f
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipArray(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipArray_t

.. py:data:: hipArray_const_t

.. py:class:: hipArray_Format

   Bases: :py:obj:`enum.IntEnum`


   HIP array format
       


   .. py:attribute:: HIP_AD_FORMAT_UNSIGNED_INT8
      :type:  int


   .. py:attribute:: HIP_AD_FORMAT_UNSIGNED_INT16
      :type:  int


   .. py:attribute:: HIP_AD_FORMAT_UNSIGNED_INT32
      :type:  int


   .. py:attribute:: HIP_AD_FORMAT_SIGNED_INT8
      :type:  int


   .. py:attribute:: HIP_AD_FORMAT_SIGNED_INT16
      :type:  int


   .. py:attribute:: HIP_AD_FORMAT_SIGNED_INT32
      :type:  int


   .. py:attribute:: HIP_AD_FORMAT_HALF
      :type:  int


   .. py:attribute:: HIP_AD_FORMAT_FLOAT
      :type:  int


.. py:class:: HIP_ARRAY_DESCRIPTOR(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP array descriptor
       


   .. py:attribute:: Width
      :type:  Any


   .. py:attribute:: Height
      :type:  Any


   .. py:attribute:: Format
      :type:  Any


   .. py:attribute:: NumChannels
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_ARRAY3D_DESCRIPTOR(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP 3D array descriptor
       


   .. py:attribute:: Width
      :type:  Any


   .. py:attribute:: Height
      :type:  Any


   .. py:attribute:: Depth
      :type:  Any


   .. py:attribute:: Format
      :type:  Any


   .. py:attribute:: NumChannels
      :type:  Any


   .. py:attribute:: Flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hip_Memcpy2D(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP 2D memory copy parameters
       


   .. py:attribute:: srcXInBytes
      :type:  Any


   .. py:attribute:: srcY
      :type:  Any


   .. py:attribute:: srcMemoryType
      :type:  Any


   .. py:attribute:: srcHost
      :type:  Any


   .. py:attribute:: srcDevice
      :type:  Any


   .. py:attribute:: srcArray
      :type:  Any


   .. py:attribute:: srcPitch
      :type:  Any


   .. py:attribute:: dstXInBytes
      :type:  Any


   .. py:attribute:: dstY
      :type:  Any


   .. py:attribute:: dstMemoryType
      :type:  Any


   .. py:attribute:: dstHost
      :type:  Any


   .. py:attribute:: dstDevice
      :type:  Any


   .. py:attribute:: dstArray
      :type:  Any


   .. py:attribute:: dstPitch
      :type:  Any


   .. py:attribute:: WidthInBytes
      :type:  Any


   .. py:attribute:: Height
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMipmappedArray(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP mipmapped array
       


   .. py:attribute:: data
      :type:  Any


   .. py:attribute:: desc
      :type:  Any


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: width
      :type:  Any


   .. py:attribute:: height
      :type:  Any


   .. py:attribute:: depth
      :type:  Any


   .. py:attribute:: min_mipmap_level
      :type:  Any


   .. py:attribute:: max_mipmap_level
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: format
      :type:  Any


   .. py:attribute:: num_channels
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipMipmappedArray_t

.. py:data:: hipmipmappedArray

.. py:data:: hipMipmappedArray_const_t

.. py:class:: hipResourceType

   Bases: :py:obj:`enum.IntEnum`


   HIP resource types
       


   .. py:attribute:: hipResourceTypeArray
      :type:  int


   .. py:attribute:: hipResourceTypeMipmappedArray
      :type:  int


   .. py:attribute:: hipResourceTypeLinear
      :type:  int


   .. py:attribute:: hipResourceTypePitch2D
      :type:  int


.. py:class:: HIPresourcetype_enum

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIP_RESOURCE_TYPE_ARRAY
      :type:  int


   .. py:attribute:: HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
      :type:  int


   .. py:attribute:: HIP_RESOURCE_TYPE_LINEAR
      :type:  int


   .. py:attribute:: HIP_RESOURCE_TYPE_PITCH2D
      :type:  int


.. py:data:: HIPresourcetype

.. py:data:: hipResourcetype

.. py:class:: HIPaddress_mode_enum

   Bases: :py:obj:`enum.IntEnum`


   HIP texture address modes
       


   .. py:attribute:: HIP_TR_ADDRESS_MODE_WRAP
      :type:  int


   .. py:attribute:: HIP_TR_ADDRESS_MODE_CLAMP
      :type:  int


   .. py:attribute:: HIP_TR_ADDRESS_MODE_MIRROR
      :type:  int


   .. py:attribute:: HIP_TR_ADDRESS_MODE_BORDER
      :type:  int


.. py:data:: HIPaddress_mode

.. py:class:: HIPfilter_mode_enum

   Bases: :py:obj:`enum.IntEnum`


   HIP filter modes
       


   .. py:attribute:: HIP_TR_FILTER_MODE_POINT
      :type:  int


   .. py:attribute:: HIP_TR_FILTER_MODE_LINEAR
      :type:  int


.. py:data:: HIPfilter_mode

.. py:class:: HIP_TEXTURE_DESC_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP texture descriptor
       


   .. py:attribute:: filterMode
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: maxAnisotropy
      :type:  Any


   .. py:attribute:: mipmapFilterMode
      :type:  Any


   .. py:attribute:: mipmapLevelBias
      :type:  Any


   .. py:attribute:: minMipmapLevelClamp
      :type:  Any


   .. py:attribute:: maxMipmapLevelClamp
      :type:  Any


   .. py:attribute:: borderColor
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: HIP_TEXTURE_DESC

.. py:class:: hipResourceViewFormat

   Bases: :py:obj:`enum.IntEnum`


   HIP texture resource view formats
       


   .. py:attribute:: hipResViewFormatNone
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedChar1
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedChar2
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedChar4
      :type:  int


   .. py:attribute:: hipResViewFormatSignedChar1
      :type:  int


   .. py:attribute:: hipResViewFormatSignedChar2
      :type:  int


   .. py:attribute:: hipResViewFormatSignedChar4
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedShort1
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedShort2
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedShort4
      :type:  int


   .. py:attribute:: hipResViewFormatSignedShort1
      :type:  int


   .. py:attribute:: hipResViewFormatSignedShort2
      :type:  int


   .. py:attribute:: hipResViewFormatSignedShort4
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedInt1
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedInt2
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedInt4
      :type:  int


   .. py:attribute:: hipResViewFormatSignedInt1
      :type:  int


   .. py:attribute:: hipResViewFormatSignedInt2
      :type:  int


   .. py:attribute:: hipResViewFormatSignedInt4
      :type:  int


   .. py:attribute:: hipResViewFormatHalf1
      :type:  int


   .. py:attribute:: hipResViewFormatHalf2
      :type:  int


   .. py:attribute:: hipResViewFormatHalf4
      :type:  int


   .. py:attribute:: hipResViewFormatFloat1
      :type:  int


   .. py:attribute:: hipResViewFormatFloat2
      :type:  int


   .. py:attribute:: hipResViewFormatFloat4
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedBlockCompressed1
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedBlockCompressed2
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedBlockCompressed3
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedBlockCompressed4
      :type:  int


   .. py:attribute:: hipResViewFormatSignedBlockCompressed4
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedBlockCompressed5
      :type:  int


   .. py:attribute:: hipResViewFormatSignedBlockCompressed5
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedBlockCompressed6H
      :type:  int


   .. py:attribute:: hipResViewFormatSignedBlockCompressed6H
      :type:  int


   .. py:attribute:: hipResViewFormatUnsignedBlockCompressed7
      :type:  int


.. py:class:: HIPresourceViewFormat_enum

   Bases: :py:obj:`enum.IntEnum`


   HIP texture resource view formats
       


   .. py:attribute:: HIP_RES_VIEW_FORMAT_NONE
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UINT_1X8
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UINT_2X8
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UINT_4X8
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SINT_1X8
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SINT_2X8
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SINT_4X8
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UINT_1X16
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UINT_2X16
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UINT_4X16
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SINT_1X16
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SINT_2X16
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SINT_4X16
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UINT_1X32
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UINT_2X32
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UINT_4X32
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SINT_1X32
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SINT_2X32
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SINT_4X32
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_FLOAT_1X16
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_FLOAT_2X16
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_FLOAT_4X16
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_FLOAT_1X32
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_FLOAT_2X32
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_FLOAT_4X32
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SIGNED_BC4
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SIGNED_BC5
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_SIGNED_BC6H
      :type:  int


   .. py:attribute:: HIP_RES_VIEW_FORMAT_UNSIGNED_BC7
      :type:  int


.. py:data:: HIPresourceViewFormat

.. py:class:: hipResourceDesc_union_0_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: array
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipResourceDesc_union_0_struct_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: mipmap
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipResourceDesc_union_0_struct_2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: devPtr
      :type:  Any


   .. py:attribute:: desc
      :type:  Any


   .. py:attribute:: sizeInBytes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipResourceDesc_union_0_struct_3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: devPtr
      :type:  Any


   .. py:attribute:: desc
      :type:  Any


   .. py:attribute:: width
      :type:  Any


   .. py:attribute:: height
      :type:  Any


   .. py:attribute:: pitchInBytes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipResourceDesc_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: array
      :type:  Any


   .. py:attribute:: mipmap
      :type:  Any


   .. py:attribute:: linear
      :type:  Any


   .. py:attribute:: pitch2D
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipResourceDesc(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP resource descriptor
       


   .. py:attribute:: resType
      :type:  Any


   .. py:attribute:: res
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_RESOURCE_DESC_st_union_0_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: hArray
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_RESOURCE_DESC_st_union_0_struct_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: hMipmappedArray
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_RESOURCE_DESC_st_union_0_struct_2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: devPtr
      :type:  Any


   .. py:attribute:: format
      :type:  Any


   .. py:attribute:: numChannels
      :type:  Any


   .. py:attribute:: sizeInBytes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_RESOURCE_DESC_st_union_0_struct_3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: devPtr
      :type:  Any


   .. py:attribute:: format
      :type:  Any


   .. py:attribute:: numChannels
      :type:  Any


   .. py:attribute:: width
      :type:  Any


   .. py:attribute:: height
      :type:  Any


   .. py:attribute:: pitchInBytes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_RESOURCE_DESC_st_union_0_struct_4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_RESOURCE_DESC_st_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: array
      :type:  Any


   .. py:attribute:: mipmap
      :type:  Any


   .. py:attribute:: linear
      :type:  Any


   .. py:attribute:: pitch2D
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_RESOURCE_DESC_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP resource view descriptor struct
       


   .. py:attribute:: resType
      :type:  Any


   .. py:attribute:: res
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: HIP_RESOURCE_DESC

.. py:class:: hipResourceViewDesc(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP resource view descriptor
       


   .. py:attribute:: format
      :type:  Any


   .. py:attribute:: width
      :type:  Any


   .. py:attribute:: height
      :type:  Any


   .. py:attribute:: depth
      :type:  Any


   .. py:attribute:: firstMipmapLevel
      :type:  Any


   .. py:attribute:: lastMipmapLevel
      :type:  Any


   .. py:attribute:: firstLayer
      :type:  Any


   .. py:attribute:: lastLayer
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_RESOURCE_VIEW_DESC_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Resource view descriptor
       


   .. py:attribute:: format
      :type:  Any


   .. py:attribute:: width
      :type:  Any


   .. py:attribute:: height
      :type:  Any


   .. py:attribute:: depth
      :type:  Any


   .. py:attribute:: firstMipmapLevel
      :type:  Any


   .. py:attribute:: lastMipmapLevel
      :type:  Any


   .. py:attribute:: firstLayer
      :type:  Any


   .. py:attribute:: lastLayer
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: HIP_RESOURCE_VIEW_DESC

.. py:class:: hipMemcpyKind

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipMemcpyHostToHost
      :type:  int


   .. py:attribute:: hipMemcpyHostToDevice
      :type:  int


   .. py:attribute:: hipMemcpyDeviceToHost
      :type:  int


   .. py:attribute:: hipMemcpyDeviceToDevice
      :type:  int


   .. py:attribute:: hipMemcpyDefault
      :type:  int


   .. py:attribute:: hipMemcpyDeviceToDeviceNoCU
      :type:  int


.. py:class:: hipPitchedPtr(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP pithed pointer
       


   .. py:attribute:: ptr
      :type:  Any


   .. py:attribute:: pitch
      :type:  Any


   .. py:attribute:: xsize
      :type:  Any


   .. py:attribute:: ysize
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExtent(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP extent
       


   .. py:attribute:: width
      :type:  Any


   .. py:attribute:: height
      :type:  Any


   .. py:attribute:: depth
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipPos(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP position
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpy3DParms(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP 3D memory copy parameters
       


   .. py:attribute:: srcArray
      :type:  Any


   .. py:attribute:: srcPos
      :type:  Any


   .. py:attribute:: srcPtr
      :type:  Any


   .. py:attribute:: dstArray
      :type:  Any


   .. py:attribute:: dstPos
      :type:  Any


   .. py:attribute:: dstPtr
      :type:  Any


   .. py:attribute:: extent
      :type:  Any


   .. py:attribute:: kind
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: HIP_MEMCPY3D(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP 3D memory copy
       


   .. py:attribute:: srcXInBytes
      :type:  Any


   .. py:attribute:: srcY
      :type:  Any


   .. py:attribute:: srcZ
      :type:  Any


   .. py:attribute:: srcLOD
      :type:  Any


   .. py:attribute:: srcMemoryType
      :type:  Any


   .. py:attribute:: srcHost
      :type:  Any


   .. py:attribute:: srcDevice
      :type:  Any


   .. py:attribute:: srcArray
      :type:  Any


   .. py:attribute:: srcPitch
      :type:  Any


   .. py:attribute:: srcHeight
      :type:  Any


   .. py:attribute:: dstXInBytes
      :type:  Any


   .. py:attribute:: dstY
      :type:  Any


   .. py:attribute:: dstZ
      :type:  Any


   .. py:attribute:: dstLOD
      :type:  Any


   .. py:attribute:: dstMemoryType
      :type:  Any


   .. py:attribute:: dstHost
      :type:  Any


   .. py:attribute:: dstDevice
      :type:  Any


   .. py:attribute:: dstArray
      :type:  Any


   .. py:attribute:: dstPitch
      :type:  Any


   .. py:attribute:: dstHeight
      :type:  Any


   .. py:attribute:: WidthInBytes
      :type:  Any


   .. py:attribute:: Height
      :type:  Any


   .. py:attribute:: Depth
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemLocationType

   Bases: :py:obj:`enum.IntEnum`


   Specifies the type of location
       


   .. py:attribute:: hipMemLocationTypeInvalid
      :type:  int


   .. py:attribute:: hipMemLocationTypeNone
      :type:  int


   .. py:attribute:: hipMemLocationTypeDevice
      :type:  int


   .. py:attribute:: hipMemLocationTypeHost
      :type:  int


   .. py:attribute:: hipMemLocationTypeHostNuma
      :type:  int


   .. py:attribute:: hipMemLocationTypeHostNumaCurrent
      :type:  int


.. py:class:: hipMemLocation(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Specifies a memory location.

   To specify a gpu, set type = ``hipMemLocationTypeDevice`` and set id = the gpu's device ID


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: id
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpyFlags

   Bases: :py:obj:`enum.IntEnum`


   Flags to specify for copies within a batch. Used with hipMemcpyBatchAsync
       


   .. py:attribute:: hipMemcpyFlagDefault
      :type:  int


   .. py:attribute:: hipMemcpyFlagPreferOverlapWithCompute
      :type:  int


   .. py:attribute:: hipMemcpyFlagExtPreferCE
      :type:  int


   .. py:attribute:: hipMemcpyFlagExtOpSwap
      :type:  int


   .. py:attribute:: hipMemcpyFlagExtOpIndirectSrc
      :type:  int


   .. py:attribute:: hipMemcpyFlagExtOpIndirectDst
      :type:  int


.. py:class:: hipMemcpySrcAccessOrder

   Bases: :py:obj:`enum.IntEnum`


   Flags to specify order in which source pointer is accessed by Batch memcpy
       


   .. py:attribute:: hipMemcpySrcAccessOrderInvalid
      :type:  int


   .. py:attribute:: hipMemcpySrcAccessOrderStream
      :type:  int


   .. py:attribute:: hipMemcpySrcAccessOrderDuringApiCall
      :type:  int


   .. py:attribute:: hipMemcpySrcAccessOrderAny
      :type:  int


   .. py:attribute:: hipMemcpySrcAccessOrderMax
      :type:  int


.. py:class:: hipMemcpyAttributes(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Attributes for copies within a batch.
       


   .. py:attribute:: srcAccessOrder
      :type:  Any


   .. py:attribute:: srcLocHint
      :type:  Any


   .. py:attribute:: dstLocHint
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpy3DOperandType

   Bases: :py:obj:`enum.IntEnum`


   Operand types for individual copies within a batch
       


   .. py:attribute:: hipMemcpyOperandTypePointer
      :type:  int


   .. py:attribute:: hipMemcpyOperandTypeArray
      :type:  int


   .. py:attribute:: hipMemcpyOperandTypeMax
      :type:  int


.. py:class:: hipOffset3D(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Struct representing offset into a hipArray_t in elements.
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpy3DOperand_union_0_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: ptr
      :type:  Any


   .. py:attribute:: rowLength
      :type:  Any


   .. py:attribute:: layerHeight
      :type:  Any


   .. py:attribute:: locHint
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpy3DOperand_union_0_struct_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: array
      :type:  Any


   .. py:attribute:: offset
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpy3DOperand_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: ptr
      :type:  Any


   .. py:attribute:: array
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpy3DOperand(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Struct representing an operand for copy with hipMemcpy3DBatchAsync.
       


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: op
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpy3DBatchOp(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP 3D Batch Op
       


   .. py:attribute:: src
      :type:  Any


   .. py:attribute:: dst
      :type:  Any


   .. py:attribute:: extent
      :type:  Any


   .. py:attribute:: srcAccessOrder
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpy3DPeerParms(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: srcArray
      :type:  Any


   .. py:attribute:: srcPos
      :type:  Any


   .. py:attribute:: srcPtr
      :type:  Any


   .. py:attribute:: srcDevice
      :type:  Any


   .. py:attribute:: dstArray
      :type:  Any


   .. py:attribute:: dstPos
      :type:  Any


   .. py:attribute:: dstPtr
      :type:  Any


   .. py:attribute:: dstDevice
      :type:  Any


   .. py:attribute:: extent
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipFunction_attribute

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_NUM_REGS
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_PTX_VERSION
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_BINARY_VERSION
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_CLUSTER_DIM_MUST_BE_SET
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
      :type:  int


   .. py:attribute:: HIP_FUNC_ATTRIBUTE_MAX
      :type:  int


.. py:class:: hipPointer_attribute

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_CONTEXT
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_HOST_POINTER
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_P2P_TOKENS
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_BUFFER_ID
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_IS_MANAGED
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_RANGE_SIZE
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_MAPPED
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
      :type:  int


   .. py:attribute:: HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
      :type:  int


.. py:class:: uchar1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: uchar2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: uchar3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: uchar4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: char1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: char2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: char3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: char4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ushort1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ushort2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ushort3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ushort4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: short1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: short2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: short3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: short4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: uint1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: uint2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: uint3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: uint4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: int1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: int2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: int3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: int4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ulong1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ulong2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ulong3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ulong4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: long1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: long2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: long3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: long4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ulonglong1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ulonglong2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ulonglong3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ulonglong4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: longlong1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: longlong2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: longlong3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: longlong4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: float1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: float2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: float3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: float4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: double1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: double2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: double3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: double4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:function:: hipCreateChannelDesc(x, y, z, w, f)

   !__LP64__

   Args:
       x (:py:obj:`~.int`):
           (undocumented)

       y (:py:obj:`~.int`):
           (undocumented)

       z (:py:obj:`~.int`):
           (undocumented)

       w (:py:obj:`~.int`):
           (undocumented)

       f (:py:obj:`~.hipChannelFormatKind`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`:
               Always returns `~.hipError_t.hipSuccess`.
       * :py:obj:`~.hipChannelFormatDesc`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       struct hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, enum hipChannelFormatKind f)


.. py:data:: hipTextureObject_t

.. py:class:: hipTextureAddressMode

   Bases: :py:obj:`enum.IntEnum`


   hip texture address modes
       


   .. py:attribute:: hipAddressModeWrap
      :type:  int


   .. py:attribute:: hipAddressModeClamp
      :type:  int


   .. py:attribute:: hipAddressModeMirror
      :type:  int


   .. py:attribute:: hipAddressModeBorder
      :type:  int


.. py:class:: hipTextureFilterMode

   Bases: :py:obj:`enum.IntEnum`


   hip texture filter modes
       


   .. py:attribute:: hipFilterModePoint
      :type:  int


   .. py:attribute:: hipFilterModeLinear
      :type:  int


.. py:class:: hipTextureReadMode

   Bases: :py:obj:`enum.IntEnum`


   hip texture read modes
       


   .. py:attribute:: hipReadModeElementType
      :type:  int


   .. py:attribute:: hipReadModeNormalizedFloat
      :type:  int


.. py:class:: textureReference(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   hip texture reference
       


   .. py:attribute:: normalized
      :type:  Any


   .. py:attribute:: readMode
      :type:  Any


   .. py:attribute:: filterMode
      :type:  Any


   .. py:attribute:: channelDesc
      :type:  Any


   .. py:attribute:: sRGB
      :type:  Any


   .. py:attribute:: maxAnisotropy
      :type:  Any


   .. py:attribute:: mipmapFilterMode
      :type:  Any


   .. py:attribute:: mipmapLevelBias
      :type:  Any


   .. py:attribute:: minMipmapLevelClamp
      :type:  Any


   .. py:attribute:: maxMipmapLevelClamp
      :type:  Any


   .. py:attribute:: textureObject
      :type:  Any


   .. py:attribute:: numChannels
      :type:  Any


   .. py:attribute:: format
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipTextureDesc(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   hip texture descriptor
       


   .. py:attribute:: filterMode
      :type:  Any


   .. py:attribute:: readMode
      :type:  Any


   .. py:attribute:: sRGB
      :type:  Any


   .. py:attribute:: borderColor
      :type:  Any


   .. py:attribute:: normalizedCoords
      :type:  Any


   .. py:attribute:: maxAnisotropy
      :type:  Any


   .. py:attribute:: mipmapFilterMode
      :type:  Any


   .. py:attribute:: mipmapLevelBias
      :type:  Any


   .. py:attribute:: minMipmapLevelClamp
      :type:  Any


   .. py:attribute:: maxMipmapLevelClamp
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipSurfaceObject_t

.. py:class:: surfaceReference(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   hip surface reference
       


   .. py:attribute:: surfaceObject
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipSurfaceBoundaryMode

   Bases: :py:obj:`enum.IntEnum`


   hip surface boundary modes
       


   .. py:attribute:: hipBoundaryModeZero
      :type:  int


   .. py:attribute:: hipBoundaryModeTrap
      :type:  int


   .. py:attribute:: hipBoundaryModeClamp
      :type:  int


.. py:class:: ihipCtx_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipCtx_t

.. py:class:: ihipExecutionCtx_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipExecutionCtx_t

.. py:class:: ihipDevResourceDesc_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipDevResourceDesc_t

.. py:class:: hipDevResourceType

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipDevResourceTypeInvalid
      :type:  int


   .. py:attribute:: hipDevResourceTypeSm
      :type:  int


   .. py:attribute:: hipDevResourceTypeWorkqueueConfig
      :type:  int


   .. py:attribute:: hipDevResourceTypeWorkqueue
      :type:  int


.. py:class:: hipDevSmResourceGroup_flags

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipDevSmResourceGroupDefault
      :type:  int


   .. py:attribute:: hipDevSmResourceGroupBackfill
      :type:  int


.. py:class:: hipDevSmResourceSplitByCount_flags

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipDevSmResourceSplitIgnoreSmCoscheduling
      :type:  int


   .. py:attribute:: hipDevSmResourceSplitMaxPotentialClusterSize
      :type:  int


.. py:class:: hipDevWorkqueueConfigScope

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipDevWorkqueueConfigScopeDeviceCtx
      :type:  int


   .. py:attribute:: hipDevWorkqueueConfigScopeGreenCtxBalanced
      :type:  int


.. py:class:: hipDevSmResource(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: smCount
      :type:  Any


   .. py:attribute:: minSmPartitionSize
      :type:  Any


   .. py:attribute:: smCoscheduledAlignment
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipDevWorkqueueConfigResource(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: device
      :type:  Any


   .. py:attribute:: wqConcurrencyLimit
      :type:  Any


   .. py:attribute:: sharingScope
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipDevWorkqueueResource(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipDevResource_st_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: sm
      :type:  Any


   .. py:attribute:: wqConfig
      :type:  Any


   .. py:attribute:: wq
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipDevResource_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: nextResource
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipDevResource

.. py:class:: hipDevSmResourceGroupParams_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: smCount
      :type:  Any


   .. py:attribute:: coscheduledSmCount
      :type:  Any


   .. py:attribute:: preferredCoscheduledSmCount
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipDevSmResourceGroupParams

.. py:class:: hipDeviceP2PAttr

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipDevP2PAttrPerformanceRank
      :type:  int


   .. py:attribute:: hipDevP2PAttrAccessSupported
      :type:  int


   .. py:attribute:: hipDevP2PAttrNativeAtomicSupported
      :type:  int


   .. py:attribute:: hipDevP2PAttrHipArrayAccessSupported
      :type:  int


.. py:class:: hipDriverEntryPointQueryResult

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipDriverEntryPointSuccess
      :type:  int


   .. py:attribute:: hipDriverEntryPointSymbolNotFound
      :type:  int


   .. py:attribute:: hipDriverEntryPointVersionNotSufficent
      :type:  int


.. py:class:: ihipStream_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipStream_t

.. py:class:: hipIpcMemHandle_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipIpcMemHandle_t

.. py:class:: hipIpcEventHandle_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipIpcEventHandle_t

.. py:class:: hipMemFabricHandle_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: data
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipMemFabricHandle_t

.. py:class:: ihipModule_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipModule_t

.. py:class:: ihipModuleSymbol_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipFunction_t

.. py:class:: ihipLinkState_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipLinkState_t

.. py:class:: ihipLibrary_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipLibrary_t

.. py:class:: ihipKernel_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipKernel_t

.. py:class:: ihipMemPoolHandle_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipMemPool_t

.. py:class:: hipFuncAttributes(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: binaryVersion
      :type:  Any


   .. py:attribute:: cacheModeCA
      :type:  Any


   .. py:attribute:: constSizeBytes
      :type:  Any


   .. py:attribute:: localSizeBytes
      :type:  Any


   .. py:attribute:: maxDynamicSharedSizeBytes
      :type:  Any


   .. py:attribute:: maxThreadsPerBlock
      :type:  Any


   .. py:attribute:: numRegs
      :type:  Any


   .. py:attribute:: preferredShmemCarveout
      :type:  Any


   .. py:attribute:: ptxVersion
      :type:  Any


   .. py:attribute:: sharedSizeBytes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ihipEvent_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipEvent_t

.. py:class:: hipLimit_t

   Bases: :py:obj:`enum.IntEnum`


   hipLimit

   Note:
       In HIP device limit-related APIs, any input limit value other than those defined in the
       enum is treated as "UnsupportedLimit" by default.


   .. py:attribute:: hipLimitStackSize
      :type:  int


   .. py:attribute:: hipLimitPrintfFifoSize
      :type:  int


   .. py:attribute:: hipLimitMallocHeapSize
      :type:  int


   .. py:attribute:: hipExtLimitScratchMin
      :type:  int


   .. py:attribute:: hipExtLimitScratchMax
      :type:  int


   .. py:attribute:: hipExtLimitScratchCurrent
      :type:  int


   .. py:attribute:: hipLimitRange
      :type:  int


.. py:class:: hipStreamBatchMemOpType

   Bases: :py:obj:`enum.IntEnum`


   Operations for hipStreamBatchMemOp
       


   .. py:attribute:: hipStreamMemOpWaitValue32
      :type:  int


   .. py:attribute:: hipStreamMemOpWriteValue32
      :type:  int


   .. py:attribute:: hipStreamMemOpWaitValue64
      :type:  int


   .. py:attribute:: hipStreamMemOpWriteValue64
      :type:  int


   .. py:attribute:: hipStreamMemOpBarrier
      :type:  int


   .. py:attribute:: hipStreamMemOpFlushRemoteWrites
      :type:  int


.. py:class:: hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: value
      :type:  Any


   .. py:attribute:: value64
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: operation
      :type:  Any


   .. py:attribute:: address
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: alias
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: value
      :type:  Any


   .. py:attribute:: value64
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: operation
      :type:  Any


   .. py:attribute:: address
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: alias
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: operation
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: operation
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipStreamBatchMemOpParams_union(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Union representing batch memory operation parameters for HIP streams.

   hipStreamBatchMemOpParams is used to specify the parameters for batch memory
   operations in a HIP stream. This union supports various operations including
   waiting for a specific value, writing a value, and different flags for wait conditions.

   The union includes fields for different types of operations defined in the
   enum hipStreamBatchMemOpType:
   - hipStreamMemOpWaitValue32:  Wait for a 32-bit value.
   - hipStreamMemOpWriteValue32: Write a 32-bit value.
   - hipStreamMemOpWaitValue64:  Wait for a 64-bit value.
   - hipStreamMemOpWriteValue64: Write a 64-bit value.

   Each operation type includes an address, the value to wait for or write, flags, and an
   optional alias that is not relevant on AMD GPUs. Flags can be used to specify different
   wait conditions such as equality, bitwise AND, greater than or equal, and bitwise NOR.

   Example usage:

   .. code-block::

      hipStreamBatchMemOpParams myArray[2];
      myArray[0].operation = hipStreamMemOpWaitValue32;
      myArray[0].waitValue.address = waitAddr1;
      myArray[0].waitValue.value = 0x1;
      myArray[0].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;

      myArray[1].operation = hipStreamMemOpWriteValue32;
      myArray[1].writeValue.address = writeAddr1;
      myArray[1].writeValue.value = 0x1;
      myArray[1].writeValue.flags = 0x0;

      result = hipStreamBatchMemOp(stream, 2, myArray, 0);


   .. py:attribute:: operation
      :type:  Any


   .. py:attribute:: waitValue
      :type:  Any


   .. py:attribute:: writeValue
      :type:  Any


   .. py:attribute:: flushRemoteWrites
      :type:  Any


   .. py:attribute:: memoryBarrier
      :type:  Any


   .. py:attribute:: pad
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipStreamBatchMemOpParams

.. py:class:: hipBatchMemOpNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Structure representing node parameters for batch memory operations in HIP graphs.

   hipBatchMemOpNodeParams is used to specify the parameters for batch memory
   operations in HIP graphs. This struct includes the context to use for the operations, the
   number of operations, and an array of hipStreamBatchMemOpParams that describe the operations.

   The structure includes the following fields:
   - ctx: The HIP context to use for the operations.
   - count: The number of operations in the paramArray.
   - paramArray: A pointer to an array of hipStreamBatchMemOpParams.
   - flags: Flags to control the node.

   Example usage:

   .. code-block::

      hipBatchMemOpNodeParams nodeParams;
      nodeParams.ctx = context;
      nodeParams.count = ARRAY_SIZE;
      nodeParams.paramArray = myArray;
      nodeParams.flags = 0;

      Pass nodeParams to a HIP graph APIs hipGraphAddBatchMemOpNode, hipGraphBatchMemOpNodeGetParams,
      hipGraphBatchMemOpNodeSetParams, hipGraphExecBatchMemOpNodeSetParams


   .. py:attribute:: ctx
      :type:  Any


   .. py:attribute:: count
      :type:  Any


   .. py:attribute:: paramArray
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemoryAdvise

   Bases: :py:obj:`enum.IntEnum`


   HIP Memory Advise values

   Note:
       This memory advise enumeration is used on Linux, not Windows.


   .. py:attribute:: hipMemAdviseSetReadMostly
      :type:  int


   .. py:attribute:: hipMemAdviseUnsetReadMostly
      :type:  int


   .. py:attribute:: hipMemAdviseSetPreferredLocation
      :type:  int


   .. py:attribute:: hipMemAdviseUnsetPreferredLocation
      :type:  int


   .. py:attribute:: hipMemAdviseSetAccessedBy
      :type:  int


   .. py:attribute:: hipMemAdviseUnsetAccessedBy
      :type:  int


   .. py:attribute:: hipMemAdviseSetCoarseGrain
      :type:  int


   .. py:attribute:: hipMemAdviseUnsetCoarseGrain
      :type:  int


.. py:class:: hipMemRangeCoherencyMode

   Bases: :py:obj:`enum.IntEnum`


   HIP Coherency Mode
       


   .. py:attribute:: hipMemRangeCoherencyModeFineGrain
      :type:  int


   .. py:attribute:: hipMemRangeCoherencyModeCoarseGrain
      :type:  int


   .. py:attribute:: hipMemRangeCoherencyModeIndeterminate
      :type:  int


.. py:class:: hipMemRangeAttribute

   Bases: :py:obj:`enum.IntEnum`


   HIP range attributes
       


   .. py:attribute:: hipMemRangeAttributeReadMostly
      :type:  int


   .. py:attribute:: hipMemRangeAttributePreferredLocation
      :type:  int


   .. py:attribute:: hipMemRangeAttributeAccessedBy
      :type:  int


   .. py:attribute:: hipMemRangeAttributeLastPrefetchLocation
      :type:  int


   .. py:attribute:: hipMemRangeAttributeCoherencyMode
      :type:  int


.. py:class:: hipMemPoolAttr

   Bases: :py:obj:`enum.IntEnum`


   HIP memory pool attributes
       


   .. py:attribute:: hipMemPoolReuseFollowEventDependencies
      :type:  int


   .. py:attribute:: hipMemPoolReuseAllowOpportunistic
      :type:  int


   .. py:attribute:: hipMemPoolReuseAllowInternalDependencies
      :type:  int


   .. py:attribute:: hipMemPoolAttrReleaseThreshold
      :type:  int


   .. py:attribute:: hipMemPoolAttrReservedMemCurrent
      :type:  int


   .. py:attribute:: hipMemPoolAttrReservedMemHigh
      :type:  int


   .. py:attribute:: hipMemPoolAttrUsedMemCurrent
      :type:  int


   .. py:attribute:: hipMemPoolAttrUsedMemHigh
      :type:  int


.. py:class:: hipMemAccessFlags

   Bases: :py:obj:`enum.IntEnum`


   Specifies the memory protection flags for mapping
       


   .. py:attribute:: hipMemAccessFlagsProtNone
      :type:  int


   .. py:attribute:: hipMemAccessFlagsProtRead
      :type:  int


   .. py:attribute:: hipMemAccessFlagsProtReadWrite
      :type:  int


.. py:class:: hipMemAccessDesc(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Memory access descriptor structure is used to specify memory access
   permissions for a virtual memory region in Virtual Memory Management API.

   This structure changes read, and write permissions for
   specific memory regions.


   .. py:attribute:: location
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemAllocationType

   Bases: :py:obj:`enum.IntEnum`


   Defines the allocation types
       


   .. py:attribute:: hipMemAllocationTypeInvalid
      :type:  int


   .. py:attribute:: hipMemAllocationTypePinned
      :type:  int


   .. py:attribute:: hipMemAllocationTypeManaged
      :type:  int


   .. py:attribute:: hipMemAllocationTypeUncached
      :type:  int


   .. py:attribute:: hipMemAllocationTypeMax
      :type:  int


.. py:class:: hipMemAllocationHandleType

   Bases: :py:obj:`enum.IntEnum`


   Flags for specifying handle types for memory pool allocations
       


   .. py:attribute:: hipMemHandleTypeNone
      :type:  int


   .. py:attribute:: hipMemHandleTypePosixFileDescriptor
      :type:  int


   .. py:attribute:: hipMemHandleTypeWin32
      :type:  int


   .. py:attribute:: hipMemHandleTypeWin32Kmt
      :type:  int


   .. py:attribute:: hipMemHandleTypeFabric
      :type:  int


.. py:class:: hipMemPoolProps(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Specifies the properties of allocations made from the pool.
       


   .. py:attribute:: allocType
      :type:  Any


   .. py:attribute:: handleTypes
      :type:  Any


   .. py:attribute:: location
      :type:  Any


   .. py:attribute:: win32SecurityAttributes
      :type:  Any


   .. py:attribute:: maxSize
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemPoolPtrExportData(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Opaque data structure for exporting a pool allocation
       


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipFuncAttribute

   Bases: :py:obj:`enum.IntEnum`


   On AMD devices and some Nvidia devices, these hints and controls are ignored.

   Warning:


   .. py:attribute:: hipFuncAttributeMaxDynamicSharedMemorySize
      :type:  int


   .. py:attribute:: hipFuncAttributePreferredSharedMemoryCarveout
      :type:  int


   .. py:attribute:: hipFuncAttributeClusterDimMustBeSet
      :type:  int


   .. py:attribute:: hipFuncAttributeRequiredClusterWidth
      :type:  int


   .. py:attribute:: hipFuncAttributeRequiredClusterHeight
      :type:  int


   .. py:attribute:: hipFuncAttributeRequiredClusterDepth
      :type:  int


   .. py:attribute:: hipFuncAttributeNonPortableClusterSizeAllowed
      :type:  int


   .. py:attribute:: hipFuncAttributeClusterSchedulingPolicyPreference
      :type:  int


   .. py:attribute:: hipFuncAttributeMax
      :type:  int


.. py:class:: hipFuncCache_t

   Bases: :py:obj:`enum.IntEnum`


   On AMD devices and some Nvidia devices, these hints and controls are ignored.

   Warning:


   .. py:attribute:: hipFuncCachePreferNone
      :type:  int


   .. py:attribute:: hipFuncCachePreferShared
      :type:  int


   .. py:attribute:: hipFuncCachePreferL1
      :type:  int


   .. py:attribute:: hipFuncCachePreferEqual
      :type:  int


.. py:class:: hipSharedMemConfig

   Bases: :py:obj:`enum.IntEnum`


   On AMD devices and some Nvidia devices, these hints and controls are ignored.

   Warning:


   .. py:attribute:: hipSharedMemBankSizeDefault
      :type:  int


   .. py:attribute:: hipSharedMemBankSizeFourByte
      :type:  int


   .. py:attribute:: hipSharedMemBankSizeEightByte
      :type:  int


.. py:class:: dim3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Struct for data in 3D
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipLaunchParams_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   struct hipLaunchParams_t
       


   .. py:attribute:: func
      :type:  Any


   .. py:attribute:: gridDim
      :type:  Any


   .. py:attribute:: blockDim
      :type:  Any


   .. py:attribute:: args
      :type:  Any


   .. py:attribute:: sharedMem
      :type:  Any


   .. py:attribute:: stream
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipLaunchParams

.. py:class:: hipFunctionLaunchParams_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   struct hipFunctionLaunchParams_t
       


   .. py:attribute:: function
      :type:  Any


   .. py:attribute:: gridDimX
      :type:  Any


   .. py:attribute:: gridDimY
      :type:  Any


   .. py:attribute:: gridDimZ
      :type:  Any


   .. py:attribute:: blockDimX
      :type:  Any


   .. py:attribute:: blockDimY
      :type:  Any


   .. py:attribute:: blockDimZ
      :type:  Any


   .. py:attribute:: sharedMemBytes
      :type:  Any


   .. py:attribute:: hStream
      :type:  Any


   .. py:attribute:: kernelParams
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipFunctionLaunchParams

.. py:class:: hipExternalMemoryHandleType_enum

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipExternalMemoryHandleTypeOpaqueFd
      :type:  int


   .. py:attribute:: hipExternalMemoryHandleTypeOpaqueWin32
      :type:  int


   .. py:attribute:: hipExternalMemoryHandleTypeOpaqueWin32Kmt
      :type:  int


   .. py:attribute:: hipExternalMemoryHandleTypeD3D12Heap
      :type:  int


   .. py:attribute:: hipExternalMemoryHandleTypeD3D12Resource
      :type:  int


   .. py:attribute:: hipExternalMemoryHandleTypeD3D11Resource
      :type:  int


   .. py:attribute:: hipExternalMemoryHandleTypeD3D11ResourceKmt
      :type:  int


   .. py:attribute:: hipExternalMemoryHandleTypeNvSciBuf
      :type:  int


.. py:data:: hipExternalMemoryHandleType

.. py:class:: hipExternalMemoryHandleDesc_st_union_0_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: handle
      :type:  Any


   .. py:attribute:: name
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalMemoryHandleDesc_st_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: fd
      :type:  Any


   .. py:attribute:: win32
      :type:  Any


   .. py:attribute:: nvSciBufObject
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalMemoryHandleDesc_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: handle
      :type:  Any


   .. py:attribute:: size
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipExternalMemoryHandleDesc

.. py:class:: hipExternalMemoryBufferDesc_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: offset
      :type:  Any


   .. py:attribute:: size
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipExternalMemoryBufferDesc

.. py:class:: hipExternalMemoryMipmappedArrayDesc_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: offset
      :type:  Any


   .. py:attribute:: formatDesc
      :type:  Any


   .. py:attribute:: extent
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: numLevels
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipExternalMemoryMipmappedArrayDesc

.. py:class:: hipExternalSemaphoreHandleType_enum

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipExternalSemaphoreHandleTypeOpaqueFd
      :type:  int


   .. py:attribute:: hipExternalSemaphoreHandleTypeOpaqueWin32
      :type:  int


   .. py:attribute:: hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
      :type:  int


   .. py:attribute:: hipExternalSemaphoreHandleTypeD3D12Fence
      :type:  int


   .. py:attribute:: hipExternalSemaphoreHandleTypeD3D11Fence
      :type:  int


   .. py:attribute:: hipExternalSemaphoreHandleTypeNvSciSync
      :type:  int


   .. py:attribute:: hipExternalSemaphoreHandleTypeKeyedMutex
      :type:  int


   .. py:attribute:: hipExternalSemaphoreHandleTypeKeyedMutexKmt
      :type:  int


   .. py:attribute:: hipExternalSemaphoreHandleTypeTimelineSemaphoreFd
      :type:  int


   .. py:attribute:: hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32
      :type:  int


.. py:data:: hipExternalSemaphoreHandleType

.. py:class:: hipExternalSemaphoreHandleDesc_st_union_0_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: handle
      :type:  Any


   .. py:attribute:: name
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreHandleDesc_st_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: fd
      :type:  Any


   .. py:attribute:: win32
      :type:  Any


   .. py:attribute:: NvSciSyncObj
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreHandleDesc_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: handle
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipExternalSemaphoreHandleDesc

.. py:class:: hipExternalSemaphoreSignalParams_st_struct_0_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: value
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreSignalParams_st_struct_0_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: fence
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreSignalParams_st_struct_0_struct_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: key
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreSignalParams_st_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: fence
      :type:  Any


   .. py:attribute:: nvSciSync
      :type:  Any


   .. py:attribute:: keyedMutex
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreSignalParams_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: params
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipExternalSemaphoreSignalParams

.. py:class:: hipExternalSemaphoreWaitParams_st_struct_0_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: value
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreWaitParams_st_struct_0_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: fence
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreWaitParams_st_struct_0_struct_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: key
      :type:  Any


   .. py:attribute:: timeoutMs
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreWaitParams_st_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: fence
      :type:  Any


   .. py:attribute:: nvSciSync
      :type:  Any


   .. py:attribute:: keyedMutex
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreWaitParams_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   External semaphore wait parameters, compatible with driver type
       


   .. py:attribute:: params
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipExternalSemaphoreWaitParams

.. py:class:: hipGraphicsRegisterFlags

   Bases: :py:obj:`enum.IntEnum`


   HIP Access falgs for Interop resources.
       


   .. py:attribute:: hipGraphicsRegisterFlagsNone
      :type:  int


   .. py:attribute:: hipGraphicsRegisterFlagsReadOnly
      :type:  int


   .. py:attribute:: hipGraphicsRegisterFlagsWriteDiscard
      :type:  int


   .. py:attribute:: hipGraphicsRegisterFlagsSurfaceLoadStore
      :type:  int


   .. py:attribute:: hipGraphicsRegisterFlagsTextureGather
      :type:  int


.. py:data:: hipGraphicsResource

.. py:data:: hipGraphicsResource_t

.. py:class:: ihipGraph(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipGraph_t

.. py:class:: hipGraphNode(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipGraphNode_t

.. py:class:: hipGraphExec(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipGraphExec_t

.. py:class:: hipUserObject(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipUserObject_t

.. py:class:: hipGraphNodeType

   Bases: :py:obj:`enum.IntEnum`


   hipGraphNodeType
       


   .. py:attribute:: hipGraphNodeTypeKernel
      :type:  int


   .. py:attribute:: hipGraphNodeTypeMemcpy
      :type:  int


   .. py:attribute:: hipGraphNodeTypeMemset
      :type:  int


   .. py:attribute:: hipGraphNodeTypeHost
      :type:  int


   .. py:attribute:: hipGraphNodeTypeGraph
      :type:  int


   .. py:attribute:: hipGraphNodeTypeEmpty
      :type:  int


   .. py:attribute:: hipGraphNodeTypeWaitEvent
      :type:  int


   .. py:attribute:: hipGraphNodeTypeEventRecord
      :type:  int


   .. py:attribute:: hipGraphNodeTypeExtSemaphoreSignal
      :type:  int


   .. py:attribute:: hipGraphNodeTypeExtSemaphoreWait
      :type:  int


   .. py:attribute:: hipGraphNodeTypeMemAlloc
      :type:  int


   .. py:attribute:: hipGraphNodeTypeMemFree
      :type:  int


   .. py:attribute:: hipGraphNodeTypeMemcpyFromSymbol
      :type:  int


   .. py:attribute:: hipGraphNodeTypeMemcpyToSymbol
      :type:  int


   .. py:attribute:: hipGraphNodeTypeBatchMemOp
      :type:  int


   .. py:attribute:: hipGraphNodeTypeCount
      :type:  int


.. py:class:: hipHostFn_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:class:: hipHostNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: fn
      :type:  Any


   .. py:attribute:: userData
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipKernelNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: blockDim
      :type:  Any


   .. py:attribute:: extra
      :type:  Any


   .. py:attribute:: func
      :type:  Any


   .. py:attribute:: gridDim
      :type:  Any


   .. py:attribute:: kernelParams
      :type:  Any


   .. py:attribute:: sharedMemBytes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemsetParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: dst
      :type:  Any


   .. py:attribute:: elementSize
      :type:  Any


   .. py:attribute:: height
      :type:  Any


   .. py:attribute:: pitch
      :type:  Any


   .. py:attribute:: value
      :type:  Any


   .. py:attribute:: width
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemAllocNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: poolProps
      :type:  Any


   .. py:attribute:: accessDescs
      :type:  Any


   .. py:attribute:: accessDescCount
      :type:  Any


   .. py:attribute:: bytesize
      :type:  Any


   .. py:attribute:: dptr
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipAccessProperty

   Bases: :py:obj:`enum.IntEnum`


   Specifies performance hint with hipAccessPolicyWindow
       


   .. py:attribute:: hipAccessPropertyNormal
      :type:  int


   .. py:attribute:: hipAccessPropertyStreaming
      :type:  int


   .. py:attribute:: hipAccessPropertyPersisting
      :type:  int


.. py:class:: hipAccessPolicyWindow(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Specifies access policy for a window, a contiguous extent of memory
   beginning at base_ptr and ending at base_ptr + num_bytes.


   .. py:attribute:: base_ptr
      :type:  Any


   .. py:attribute:: hitProp
      :type:  Any


   .. py:attribute:: hitRatio
      :type:  Any


   .. py:attribute:: missProp
      :type:  Any


   .. py:attribute:: num_bytes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipLaunchMemSyncDomainMap(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Memory Synchronization Domain map
       


   .. py:attribute:: default_
      :type:  Any


   .. py:attribute:: remote
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipLaunchMemSyncDomain

   Bases: :py:obj:`enum.IntEnum`


   Memory Synchronization Domain
       


   .. py:attribute:: hipLaunchMemSyncDomainDefault
      :type:  int


   .. py:attribute:: hipLaunchMemSyncDomainRemote
      :type:  int


.. py:class:: hipSynchronizationPolicy

   Bases: :py:obj:`enum.IntEnum`


   Stream Synchronization Policy.

   Can be set with hipStreamSetAttribute


   .. py:attribute:: hipSyncPolicyAuto
      :type:  int


   .. py:attribute:: hipSyncPolicySpin
      :type:  int


   .. py:attribute:: hipSyncPolicyYield
      :type:  int


   .. py:attribute:: hipSyncPolicyBlockingSync
      :type:  int


.. py:class:: hipClusterSchedulingPolicy

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipClusterSchedulingPolicyDefault
      :type:  int


   .. py:attribute:: hipClusterSchedulingPolicySpread
      :type:  int


   .. py:attribute:: hipClusterSchedulingPolicyLoadBalancing
      :type:  int


.. py:class:: hipExtDynDataPrefetchTemporal

   Bases: :py:obj:`enum.IntEnum`


   Temporal locality hint for dynamic data prefetch.
       


   .. py:attribute:: hipExtDynDataPrefetchTemporalRegular
      :type:  int


   .. py:attribute:: hipExtDynDataPrefetchTemporalHigh
      :type:  int


.. py:class:: hipExtDynDataPrefetchRegion(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Describes one 2D memory region to prefetch into L2 cache.

   ``address`` must be aligned to the device's L2 cache line size.
   ``width`` is measured in bytes and must be a multiple of the cache line size.
   ``height`` is the number of rows to prefetch.
   ``stride`` is the byte stride between the start of consecutive rows.


   .. py:attribute:: address
      :type:  Any


   .. py:attribute:: stride
      :type:  Any


   .. py:attribute:: width
      :type:  Any


   .. py:attribute:: height
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExtDynDataPrefetchConfig(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Configuration for dynamic data prefetch.

   Pointed to by a launch attribute via ``hipLaunchAttributeExtDynDataPrefetch`` .
   ``numRegions`` must not exceed the device limit queried via
   ``hipDeviceAttributeMaxDynDataPrefetchRegions`` (use ``hipDeviceGetAttribute`` ).

   Cooperative prefetch is always enabled internally; the user only controls
   the temporal policy.


   .. py:attribute:: numRegions
      :type:  Any


   .. py:attribute:: temporal
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipLaunchAttributeID

   Bases: :py:obj:`enum.IntEnum`


   Launch Attribute ID
       


   .. py:attribute:: hipLaunchAttributeIgnore
      :type:  int


   .. py:attribute:: hipLaunchAttributeAccessPolicyWindow
      :type:  int


   .. py:attribute:: hipLaunchAttributeCooperative
      :type:  int


   .. py:attribute:: hipLaunchAttributeSynchronizationPolicy
      :type:  int


   .. py:attribute:: hipLaunchAttributeClusterDimension
      :type:  int


   .. py:attribute:: hipLaunchAttributeClusterSchedulingPolicyPreference
      :type:  int


   .. py:attribute:: hipLaunchAttributePriority
      :type:  int


   .. py:attribute:: hipLaunchAttributeMemSyncDomainMap
      :type:  int


   .. py:attribute:: hipLaunchAttributeMemSyncDomain
      :type:  int


   .. py:attribute:: hipLaunchAttributeExtDynDataPrefetch
      :type:  int


   .. py:attribute:: hipLaunchAttributeMax
      :type:  int


.. py:class:: hipLaunchAttributeValue_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Specifies the desired cluster dimensions for a kernel launch.

   This opaque type is used as the value for the launch attribute
   ::hipLaunchAttributeClusterDimension. It defines the dimensions of the
   compute cluster in terms of blocks, where each field must evenly divide
   the corresponding grid dimension:

    - ``x:`` Number of blocks along the X-axis.
    - ``y:`` Number of blocks along the Y-axis.
    - ``z:`` Number of blocks along the Z-axis.


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipLaunchAttributeValue(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Launch Attribute Value
       


   .. py:attribute:: pad
      :type:  Any


   .. py:attribute:: accessPolicyWindow
      :type:  Any


   .. py:attribute:: cooperative
      :type:  Any


   .. py:attribute:: priority
      :type:  Any


   .. py:attribute:: syncPolicy
      :type:  Any


   .. py:attribute:: memSyncDomainMap
      :type:  Any


   .. py:attribute:: memSyncDomain
      :type:  Any


   .. py:attribute:: clusterDim
      :type:  Any


   .. py:attribute:: clusterSchedulingPolicyPreference
      :type:  Any


   .. py:attribute:: dynDataPrefetch
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipGraphExecUpdateResult

   Bases: :py:obj:`enum.IntEnum`


   Graph execution update result
       


   .. py:attribute:: hipGraphExecUpdateSuccess
      :type:  int


   .. py:attribute:: hipGraphExecUpdateError
      :type:  int


   .. py:attribute:: hipGraphExecUpdateErrorTopologyChanged
      :type:  int


   .. py:attribute:: hipGraphExecUpdateErrorNodeTypeChanged
      :type:  int


   .. py:attribute:: hipGraphExecUpdateErrorFunctionChanged
      :type:  int


   .. py:attribute:: hipGraphExecUpdateErrorParametersChanged
      :type:  int


   .. py:attribute:: hipGraphExecUpdateErrorNotSupported
      :type:  int


   .. py:attribute:: hipGraphExecUpdateErrorUnsupportedFunctionChange
      :type:  int


.. py:class:: hipStreamCaptureMode

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipStreamCaptureModeGlobal
      :type:  int


   .. py:attribute:: hipStreamCaptureModeThreadLocal
      :type:  int


   .. py:attribute:: hipStreamCaptureModeRelaxed
      :type:  int


.. py:class:: hipStreamCaptureStatus

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipStreamCaptureStatusNone
      :type:  int


   .. py:attribute:: hipStreamCaptureStatusActive
      :type:  int


   .. py:attribute:: hipStreamCaptureStatusInvalidated
      :type:  int


.. py:class:: hipStreamUpdateCaptureDependenciesFlags

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipStreamAddCaptureDependencies
      :type:  int


   .. py:attribute:: hipStreamSetCaptureDependencies
      :type:  int


.. py:class:: hipGraphMemAttributeType

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipGraphMemAttrUsedMemCurrent
      :type:  int


   .. py:attribute:: hipGraphMemAttrUsedMemHigh
      :type:  int


   .. py:attribute:: hipGraphMemAttrReservedMemCurrent
      :type:  int


   .. py:attribute:: hipGraphMemAttrReservedMemHigh
      :type:  int


.. py:class:: hipUserObjectFlags

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipUserObjectNoDestructorSync
      :type:  int


.. py:class:: hipUserObjectRetainFlags

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipGraphUserObjectMove
      :type:  int


.. py:class:: hipGraphInstantiateFlags

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipGraphInstantiateFlagAutoFreeOnLaunch
      :type:  int


   .. py:attribute:: hipGraphInstantiateFlagUpload
      :type:  int


   .. py:attribute:: hipGraphInstantiateFlagDeviceLaunch
      :type:  int


   .. py:attribute:: hipGraphInstantiateFlagUseNodePriority
      :type:  int


.. py:class:: hipGraphDebugDotFlags

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipGraphDebugDotFlagsVerbose
      :type:  int


   .. py:attribute:: hipGraphDebugDotFlagsKernelNodeParams
      :type:  int


   .. py:attribute:: hipGraphDebugDotFlagsMemcpyNodeParams
      :type:  int


   .. py:attribute:: hipGraphDebugDotFlagsMemsetNodeParams
      :type:  int


   .. py:attribute:: hipGraphDebugDotFlagsHostNodeParams
      :type:  int


   .. py:attribute:: hipGraphDebugDotFlagsEventNodeParams
      :type:  int


   .. py:attribute:: hipGraphDebugDotFlagsExtSemasSignalNodeParams
      :type:  int


   .. py:attribute:: hipGraphDebugDotFlagsExtSemasWaitNodeParams
      :type:  int


   .. py:attribute:: hipGraphDebugDotFlagsKernelNodeAttributes
      :type:  int


   .. py:attribute:: hipGraphDebugDotFlagsHandles
      :type:  int


.. py:class:: hipGraphInstantiateResult

   Bases: :py:obj:`enum.IntEnum`


   hipGraphInstantiateWithParams results
       


   .. py:attribute:: hipGraphInstantiateSuccess
      :type:  int


   .. py:attribute:: hipGraphInstantiateError
      :type:  int


   .. py:attribute:: hipGraphInstantiateInvalidStructure
      :type:  int


   .. py:attribute:: hipGraphInstantiateNodeOperationNotSupported
      :type:  int


   .. py:attribute:: hipGraphInstantiateMultipleDevicesNotSupported
      :type:  int


.. py:class:: hipGraphInstantiateParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Graph Instantiation parameters
       


   .. py:attribute:: errNode_out
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: result_out
      :type:  Any


   .. py:attribute:: uploadStream
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemAllocationProp_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: requestedHandleType
      :type:  Any


   .. py:attribute:: requestedHandleTypes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemAllocationProp_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: compressionType
      :type:  Any


   .. py:attribute:: gpuDirectRDMACapable
      :type:  Any


   .. py:attribute:: usage
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemAllocationProp(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Memory allocation properties
       


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: location
      :type:  Any


   .. py:attribute:: win32HandleMetaData
      :type:  Any


   .. py:attribute:: allocFlags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreSignalNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   External semaphore signal node parameters
       


   .. py:attribute:: extSemArray
      :type:  Any


   .. py:attribute:: paramsArray
      :type:  Any


   .. py:attribute:: numExtSems
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipExternalSemaphoreWaitNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   External semaphore wait node parameters
       


   .. py:attribute:: extSemArray
      :type:  Any


   .. py:attribute:: paramsArray
      :type:  Any


   .. py:attribute:: numExtSems
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ihipMemGenericAllocationHandle(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipMemGenericAllocationHandle_t

.. py:class:: hipMemAllocationGranularity_flags

   Bases: :py:obj:`enum.IntEnum`


   Flags for granularity
       


   .. py:attribute:: hipMemAllocationGranularityMinimum
      :type:  int


   .. py:attribute:: hipMemAllocationGranularityRecommended
      :type:  int


.. py:class:: hipMemHandleType

   Bases: :py:obj:`enum.IntEnum`


   Memory handle type
       


   .. py:attribute:: hipMemHandleTypeGeneric
      :type:  int


.. py:class:: hipMemOperationType

   Bases: :py:obj:`enum.IntEnum`


   Memory operation types
       


   .. py:attribute:: hipMemOperationTypeMap
      :type:  int


   .. py:attribute:: hipMemOperationTypeUnmap
      :type:  int


.. py:class:: hipArraySparseSubresourceType

   Bases: :py:obj:`enum.IntEnum`


   Subresource types for sparse arrays
       


   .. py:attribute:: hipArraySparseSubresourceTypeSparseLevel
      :type:  int


   .. py:attribute:: hipArraySparseSubresourceTypeMiptail
      :type:  int


.. py:class:: hipArrayMapInfo_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: mipmap
      :type:  Any


   .. py:attribute:: array
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipArrayMapInfo_union_1_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: level
      :type:  Any


   .. py:attribute:: layer
      :type:  Any


   .. py:attribute:: offsetX
      :type:  Any


   .. py:attribute:: offsetY
      :type:  Any


   .. py:attribute:: offsetZ
      :type:  Any


   .. py:attribute:: extentWidth
      :type:  Any


   .. py:attribute:: extentHeight
      :type:  Any


   .. py:attribute:: extentDepth
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipArrayMapInfo_union_1_struct_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: layer
      :type:  Any


   .. py:attribute:: offset
      :type:  Any


   .. py:attribute:: size
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipArrayMapInfo_union_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: sparseLevel
      :type:  Any


   .. py:attribute:: miptail
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipArrayMapInfo_union_2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: memHandle
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipArrayMapInfo(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Map info for arrays
       


   .. py:attribute:: resourceType
      :type:  Any


   .. py:attribute:: resource
      :type:  Any


   .. py:attribute:: subresourceType
      :type:  Any


   .. py:attribute:: subresource
      :type:  Any


   .. py:attribute:: memOperationType
      :type:  Any


   .. py:attribute:: memHandleType
      :type:  Any


   .. py:attribute:: memHandle
      :type:  Any


   .. py:attribute:: offset
      :type:  Any


   .. py:attribute:: deviceBitMask
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemcpyNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Memcpy node params
       


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:attribute:: copyParams
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipChildGraphNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Child graph node params
       


   .. py:attribute:: graph
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipEventWaitNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Event record node params
       


   .. py:attribute:: event
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipEventRecordNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Event record node params
       


   .. py:attribute:: event
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemFreeNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Memory free node params
       


   .. py:attribute:: dptr
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipGraphNodeParams_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: reserved1
      :type:  Any


   .. py:attribute:: kernel
      :type:  Any


   .. py:attribute:: memcpy
      :type:  Any


   .. py:attribute:: memset
      :type:  Any


   .. py:attribute:: host
      :type:  Any


   .. py:attribute:: graph
      :type:  Any


   .. py:attribute:: eventWait
      :type:  Any


   .. py:attribute:: eventRecord
      :type:  Any


   .. py:attribute:: extSemSignal
      :type:  Any


   .. py:attribute:: extSemWait
      :type:  Any


   .. py:attribute:: alloc
      :type:  Any


   .. py:attribute:: free
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipGraphNodeParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Params for different graph nodes
       


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: reserved0
      :type:  Any


   .. py:attribute:: reserved2
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipGraphDependencyType

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: hipGraphDependencyTypeDefault
      :type:  int


   .. py:attribute:: hipGraphDependencyTypeProgrammatic
      :type:  int


.. py:class:: hipGraphEdgeData(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: from_port
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:attribute:: to_port
      :type:  Any


   .. py:attribute:: type
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipLaunchAttribute_st_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: val
      :type:  Any


   .. py:attribute:: value
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipLaunchAttribute_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Used to specify custom attributes for launching kernels
       


   .. py:attribute:: id
      :type:  Any


   .. py:attribute:: pad
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipLaunchAttribute

.. py:class:: hipLaunchConfig_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP extensible launch configuration
       


   .. py:attribute:: gridDim
      :type:  Any


   .. py:attribute:: blockDim
      :type:  Any


   .. py:attribute:: dynamicSmemBytes
      :type:  Any


   .. py:attribute:: stream
      :type:  Any


   .. py:attribute:: attrs
      :type:  Any


   .. py:attribute:: numAttrs
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipLaunchConfig_t

.. py:class:: HIP_LAUNCH_CONFIG_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HIP driver extensible launch configuration
       


   .. py:attribute:: gridDimX
      :type:  Any


   .. py:attribute:: gridDimY
      :type:  Any


   .. py:attribute:: gridDimZ
      :type:  Any


   .. py:attribute:: blockDimX
      :type:  Any


   .. py:attribute:: blockDimY
      :type:  Any


   .. py:attribute:: blockDimZ
      :type:  Any


   .. py:attribute:: sharedMemBytes
      :type:  Any


   .. py:attribute:: hStream
      :type:  Any


   .. py:attribute:: attrs
      :type:  Any


   .. py:attribute:: numAttrs
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: HIP_LAUNCH_CONFIG

.. py:class:: hipArrayMemoryRequirements(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Struct representing array memory requirements.
       


   .. py:attribute:: alignment
      :type:  Any


   .. py:attribute:: size
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipMemRangeHandleType

   Bases: :py:obj:`enum.IntEnum`


   Requested handle type for address range.
       


   .. py:attribute:: hipMemRangeHandleTypeDmaBufFd
      :type:  int


   .. py:attribute:: hipMemRangeHandleTypeMax
      :type:  int


.. py:class:: hipMemRangeFlags

   Bases: :py:obj:`enum.IntEnum`


   Mem Range Flags used in hipMemGetHandleForAddressRange.
       


   .. py:attribute:: hipMemRangeFlagDmaBufMappingTypePcie
      :type:  int


   .. py:attribute:: hipMemRangeFlagsMax
      :type:  int


.. py:function:: hipInit(flags)

   Explicitly initializes the HIP runtime.

   *
   Defines the HIP API.  See the individual sections for more information.

   *  This section describes the initializtion and version functions of HIP runtime API.

   Most HIP APIs implicitly initialize the HIP runtime.
   This API provides control over the timing of the initialization.

   Note:
       Applications that use fork() should not initialize the HIP runtime
       before the fork when the child process will continue executing HIP code
       without an immediate exec(). Instead, the parent and child processes should
       initialize HIP independently after fork(). Inheriting HIP runtime state
       across fork() may lead to undefined behavior or initialization failures.

   Args:
       flags (:py:obj:`~.int`) -- *IN*:
           Initialization flag, should be zero.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipInit(unsigned int flags)


.. py:function:: hipDriverGetVersion()

   Returns the approximate HIP driver version.

   HIP driver version shows up in the format:
   HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH.

   Warning:
       The HIP driver version does not correspond to an exact CUDA driver revision.
       On AMD platform, the API returns the HIP driver version, while on NVIDIA platform, it calls
       the corresponding CUDA runtime API and returns the CUDA driver version.
       There is no mapping/correlation between HIP driver version and CUDA driver version.

   See:
       :py:obj:`~.hipRuntimeGetVersion`

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               driver version

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDriverGetVersion(int * driverVersion)


.. py:function:: hipRuntimeGetVersion()

   Returns the approximate HIP Runtime version.

   Warning:
       The version definition of HIP runtime is different from CUDA.
       On AMD platform, the function returns HIP runtime version,
       while on NVIDIA platform, it returns CUDA runtime version.
       And there is no mapping/correlation between HIP version and CUDA version.

   See:
       :py:obj:`~.hipDriverGetVersion`

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               HIP runtime version

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipRuntimeGetVersion(int * runtimeVersion)


.. py:function:: hipDeviceGet(ordinal)

   Returns a handle to a compute device

   Args:
       ordinal (:py:obj:`~.int`) -- *IN*:
           Device ordinal

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.int`:
               Handle of device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGet(hipDevice_t * device, int ordinal)


.. py:function:: hipDeviceComputeCapability(device)

   Returns the compute capability of the device

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           Device ordinal

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.int`:
               Major compute capability version number
       * :py:obj:`~.int`:
               Minor compute capability version number

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceComputeCapability(int * major, int * minor, hipDevice_t device)


.. py:function:: hipDeviceGetName(len, device)

   Returns an identifer string for the device.

   Args:
       len (:py:obj:`~.int`) -- *IN*:
           Maximum length of string to store in device name

       device (:py:obj:`~.int`) -- *IN*:
           Device ordinal

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               String of the device name

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetName(char * name, int len, hipDevice_t device)


.. py:function:: hipDeviceGetUuid(device)

   Returns an UUID for the device.[BETA]

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           device ordinal

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`,
           :py:obj:`~.hipErrorDeinitialized`
       * :py:obj:`~.hipUUID_t`:
               UUID for the device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetUuid(hipUUID * uuid, hipDevice_t device)


.. py:function:: hipDeviceGetP2PAttribute(attr, srcDevice, dstDevice)

   Returns a value for attribute of link between two devices

   Args:
       attr (:py:obj:`~.hipDeviceP2PAttr`) -- *IN*:
           enum of hipDeviceP2PAttr to query

       srcDevice (:py:obj:`~.int`) -- *IN*:
           The source device of the link

       dstDevice (:py:obj:`~.int`) -- *IN*:
           The destination device of the link

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.int`:
               Pointer of the value for the attrubute

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetP2PAttribute(int * value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice)


.. py:function:: hipDeviceGetPCIBusId(len, device)

   Returns a PCI Bus Id string for the device, overloaded to take int device ID.

   Args:
       len (:py:obj:`~.int`) -- *IN*:
           Maximum length of string

       device (:py:obj:`~.int`) -- *IN*:
           The device ordinal

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               The string of PCI Bus Id format for the device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetPCIBusId(char * pciBusId, int len, int device)


.. py:function:: hipDeviceGetByPCIBusId(pciBusId)

   Returns a handle to a compute device.

   Args:
       pciBusId (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           The string of PCI Bus Id for the device

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               The handle of the device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetByPCIBusId(int * device, const char * pciBusId)


.. py:function:: hipDeviceTotalMem(device)

   Returns the total amount of memory on the device.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           The ordinal of the device

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.int`:
               The size of memory in bytes, on the device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceTotalMem(size_t * bytes, hipDevice_t device)


.. py:function:: hipDeviceSynchronize()

   Waits on all active streams on current device

   *  This section describes the device management functions of HIP runtime API.

   When this command is invoked, the host thread gets blocked until all the commands associated
   with streams associated with the device. HIP does not support multiple blocking modes (yet!).

   See:
       :py:obj:`~.hipSetDevice`, :py:obj:`~.hipDeviceReset`

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceSynchronize()


.. py:function:: hipDeviceReset()

   The state of current device is discarded and updated to a fresh state.

   Calling this function deletes all streams created, memory allocated, kernels running, events
   created. Make sure that no other thread is using the device or streams, memory, kernels, events
   associated with the current device.

   See:
       :py:obj:`~.hipDeviceSynchronize`

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceReset()


.. py:function:: hipSetDevice(deviceId)

   Set default device to be used for subsequent hip API calls from this thread.

   Sets ``device`` as the default device for the calling host thread.  Valid device id's are 0...
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

   See:
       :py:obj:`~.hipGetDevice`, :py:obj:`~.hipGetDeviceCount`

   Args:
       deviceId (:py:obj:`~.int`) -- *IN*:
           Valid device in range 0...hipGetDeviceCount().

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorNoDevice`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipSetDevice(int deviceId)


.. py:function:: hipSetValidDevices(len)

   Set a list of devices that can be used.

   See:
       :py:obj:`~.hipGetDevice`, :py:obj:`~.hipGetDeviceCount`. :py:obj:`~.hipSetDevice`. :py:obj:`~.hipGetDeviceProperties`.
       :py:obj:`~.hipSetDeviceFlags`. :py:obj:`~.hipChooseDevice`

   Args:
       len (:py:obj:`~.int`) -- *IN*:
           Number of devices in specified list

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               List of devices to try

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipSetValidDevices(int * device_arr, int len)


.. py:function:: hipGetDevice()

   Return the default device id for the calling host thread.

   HIP maintains an default device for each thread using thread-local-storage.
   This device is used implicitly for HIP runtime APIs called by this thread.
   hipGetDevice returns in * ``device`` the default device for the calling host thread.

   See:
       :py:obj:`~.hipSetDevice`, :py:obj:`~.hipGetDevicesizeBytes`

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               *device is written with the default device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetDevice(int * deviceId)


.. py:function:: hipGetDeviceCount()

   Return number of compute-capable devices.

   Returns in ``*count`` the number of devices that have ability to run compute commands.  If there
   are no such devices, then ``hipGetDeviceCount`` will return :py:obj:`~.hipErrorNoDevice`. If 1 or more
   devices can be found, then hipGetDeviceCount returns :py:obj:`~.hipSuccess`.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNoDevice`
       * :py:obj:`~.int`:
               Returns number of compute-capable devices.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetDeviceCount(int * count)


.. py:function:: hipDeviceGetAttribute(attr, deviceId)

   Query for a specific device attribute.

   Args:
       attr (:py:obj:`~.hipDeviceAttribute_t`) -- *IN*:
           attribute to query

       deviceId (:py:obj:`~.int`) -- *IN*:
           which device to query for information

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               pointer to value to return

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetAttribute(int * pi, hipDeviceAttribute_t attr, int deviceId)


.. py:function:: hipDeviceGetDefaultMemPool(device)

   Returns the default memory pool of the specified device

   See:
       :py:obj:`~.hipDeviceGetDefaultMemPool`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolSetAccess`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           Device index for query the default memory pool

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.ihipMemPoolHandle_t`:
               Default memory pool to return

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t * mem_pool, int device)


.. py:function:: hipDeviceSetMemPool(device, mem_pool)

   Sets the current memory pool of a device

   The memory pool must be local to the specified device.
   ``hipMallocAsync`` allocates from the current mempool of the provided stream's device.
   By default, a device's current memory pool is its default memory pool.

   Note:
       Use ``hipMallocFromPoolAsync`` for asynchronous memory allocations from a device
       different than the one the stream runs on.

   See:
       :py:obj:`~.hipDeviceGetDefaultMemPool`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolSetAccess`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           Device index for the update

       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           Memory pool for update as the current on the specified device

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool)


.. py:function:: hipDeviceGetMemPool(device)

   Gets the current memory pool for the specified device

   Returns the last pool provided to ``hipDeviceSetMemPool`` for this device
   or the device's default memory pool if ``hipDeviceSetMemPool`` has never been called.
   By default the current mempool is the default mempool for a device,
   otherwise the returned pool must have been set with ``hipDeviceSetMemPool.``

   See:
       :py:obj:`~.hipDeviceGetDefaultMemPool`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolSetAccess`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           Device index to query the current memory pool

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.ihipMemPoolHandle_t`:
               Current memory pool on the specified device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetMemPool(hipMemPool_t * mem_pool, int device)


.. py:function:: hipGetDeviceProperties(deviceId)

   Returns device properties.

   Bug:
       HIP-Clang always returns 0 for maxThreadsPerMultiProcessor

   Bug:
       HIP-Clang always returns 0 for regsPerBlock

   Bug:
       HIP-Clang always returns 0 for l2CacheSize

   Populates hipGetDeviceProperties with information for the specified device.

   Args:
       deviceId (:py:obj:`~.int`) -- *IN*:
           which device to query for information

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.hipDeviceProp_t`:
               written with device properties

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600 * prop, int deviceId)


.. py:function:: hipDeviceGetTexture1DLinearMaxWidth(desc, device)

   Gets the maximum width for 1D linear textures on the specified device

   This function queries the maximum width, in elements, of 1D linear textures that can be allocated
   on the specified device. The maximum width depends on the texture element size and the hardware
   limitations of the device.

   See:
       :py:obj:`~.hipDeviceGetAttribute`, :py:obj:`~.hipMalloc`, :py:obj:`~.hipTexRefSetAddressMode`

   Args:
       desc (:py:obj:`~.hipChannelFormatDesc`/:py:obj:`~.object`) -- *IN*:
           Requested channel format

       device (:py:obj:`~.int`) -- *IN*:
           Device index to query for maximum 1D texture width

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.int`:
               Maximum width, in elements, of 1D linear textures that the device can
               support

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetTexture1DLinearMaxWidth(size_t * max_width, const hipChannelFormatDesc * desc, int device)


.. py:function:: hipDeviceSetCacheConfig(cacheConfig)

   Set L1/Shared cache partition.

   Note: AMD devices do not support reconfigurable cache. This API is not implemented
   on AMD platform. If the function is called, it will return hipErrorNotSupported.

   Args:
       cacheConfig (:py:obj:`~.hipFuncCache_t`) -- *IN*:
           Cache configuration

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig)


.. py:function:: hipDeviceGetCacheConfig()

   Get Cache configuration for a specific Device

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`
           Note: AMD devices do not support reconfigurable cache. This hint is ignored
           on these architectures.
       * :py:obj:`~.hipFuncCache_t`:
               Pointer of cache configuration

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetCacheConfig(hipFuncCache_t * cacheConfig)


.. py:function:: hipDeviceGetLimit(limit)

   Gets resource limits of current device

   The function queries the size of limit value, as required by the input enum value hipLimit_t,
   which can be either :py:obj:`~.hipLimitStackSize`, or :py:obj:`~.hipLimitMallocHeapSize`. Any other input as
   default, the function will return :py:obj:`~.hipErrorUnsupportedLimit`.

   Args:
       limit (:py:obj:`~.hipLimit_t`) -- *IN*:
           The limit to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorUnsupportedLimit`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Returns the size of the limit in bytes

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetLimit(size_t * pValue, enum hipLimit_t limit)


.. py:function:: hipDeviceSetLimit(limit, value)

   Sets resource limits of current device.

   As the input enum limit,
   :py:obj:`~.hipLimitStackSize` sets the limit value of the stack size on the current GPU device, per thread.
   The limit size can get via hipDeviceGetLimit. The size is in units of 256 dwords, up to the limit
   (128K - 16).

   :py:obj:`~.hipLimitMallocHeapSize` sets the limit value of the heap used by the malloc()/free()
   calls. For limit size, use the :py:obj:`~.hipDeviceGetLimit` API.

   Any other input as default, the funtion will return hipErrorUnsupportedLimit.

   Args:
       limit (:py:obj:`~.hipLimit_t`) -- *IN*:
           Enum of hipLimit_t to set

       value (:py:obj:`~.int`) -- *IN*:
           The size of limit value in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorUnsupportedLimit`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value)


.. py:function:: hipDeviceGetSharedMemConfig()

   Returns bank width of shared memory for current device

   Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
   ignored on those architectures.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`
       * :py:obj:`~.hipSharedMemConfig`:
               The pointer of the bank width for shared memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig * pConfig)


.. py:function:: hipGetDeviceFlags()

   Gets the flags set for current device

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Pointer of the flags

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetDeviceFlags(unsigned int * flags)


.. py:function:: hipDeviceSetSharedMemConfig(config)

   The bank width of shared memory on current device is set

   Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
   ignored on those architectures.

   Args:
       config (:py:obj:`~.hipSharedMemConfig`) -- *IN*:
           Configuration for the bank width of shared memory

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config)


.. py:function:: hipSetDeviceFlags(flags)

   The current device behavior is changed according to the flags passed.

   The schedule flags impact how HIP waits for the completion of a command running on a device.

   :py:obj:`~.hipDeviceScheduleSpin`         : HIP runtime will actively spin in the thread which submitted
   the work until the command completes.  This offers the lowest latency, but will consume a CPU
   core and may increase power.

   :py:obj:`~.hipDeviceScheduleYield`        : The HIP runtime will yield the CPU to system so that other
   tasks can use it. This may increase latency to detect the completion but will consume less
   power and is friendlier to other tasks in the system.

   :py:obj:`~.hipDeviceScheduleBlockingSync` : On ROCm platform, this is a synonym for hipDeviceScheduleYield.

   :py:obj:`~.hipDeviceScheduleAuto`         : This is the default value if the input 'flags' is zero.
   Uses a heuristic to select between Spin and Yield modes. If the number of HIP contexts is
   greater than the number of logical processors in the system, uses Spin scheduling, otherwise
   uses Yield scheduling.

   :py:obj:`~.hipDeviceMapHost`              : Allows mapping host memory. On ROCm, this is always allowed and
   the flag is ignored.

   :py:obj:`~.hipDeviceLmemResizeToMax`      : This flag is silently ignored on ROCm.

   Args:
       flags (:py:obj:`~.int`) -- *IN*:
           Flag to set on the current device

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNoDevice`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorSetOnActiveProcess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipSetDeviceFlags(unsigned int flags)


.. py:function:: hipChooseDevice(prop)

   Device which matches hipDeviceProp_t is returned

   Args:
       prop (:py:obj:`~.hipDeviceProp_t`/:py:obj:`~.object`) -- *IN*:
           Pointer of the properties

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Pointer of the device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipChooseDeviceR0600(int * device, const hipDeviceProp_tR0600 * prop)


.. py:function:: hipExtGetLinkTypeAndHopCount(device1, device2)

   Returns the link type and hop count between two devices

   Queries and returns the HSA link type and the hop count between the two specified devices.

   Args:
       device1 (:py:obj:`~.int`) -- *IN*:
           Ordinal for device1

       device2 (:py:obj:`~.int`) -- *IN*:
           Ordinal for device2

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Returns the link type (See hsa_amd_link_info_type_t) between the two
               devices
       * :py:obj:`~.int`:
               Returns the hop count between the two devices

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t * linktype, uint32_t * hopcount)


.. py:function:: hipIpcGetMemHandle(devPtr)

   Gets an interprocess memory handle for an existing device memory
            allocation

   Takes a pointer to the base of an existing device memory allocation created
   with hipMalloc and exports it for use in another process. This is a
   lightweight operation and may be called multiple times on an allocation
   without adverse effects.

   If a region of memory is freed with hipFree and a subsequent call
   to hipMalloc returns memory with the same device address,
   hipIpcGetMemHandle will return a unique handle for the
   new memory.

   Note:
       This IPC memory related feature API on Windows may behave differently from Linux.

   Args:
       devPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           - Base pointer to previously allocated device memory

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorOutOfMemory`, :py:obj:`~.hipErrorMapFailed`
       * :py:obj:`~.hipIpcMemHandle_st`:
               - Pointer to user allocated hipIpcMemHandle to return
               the handle in.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t * handle, void * devPtr)


.. py:function:: hipIpcOpenMemHandle(handle, flags)

   Opens an interprocess memory handle exported from another process
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

   Note:
       During multiple processes, using the same memory handle opened by the current context,
       there is no guarantee that the same device poiter will be returned in ``*devPtr.``
       This is diffrent from CUDA.

   Note:
       This IPC memory related feature API on Windows may behave differently from Linux.

   Args:
       handle (:py:obj:`~.hipIpcMemHandle_st`):
           - hipIpcMemHandle to open

       flags (:py:obj:`~.int`):
           - Flags for this operation. Must be specified as hipIpcMemLazyEnablePeerAccess

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidDevicePointer`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               - Returned device pointer

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipIpcOpenMemHandle(void ** devPtr, hipIpcMemHandle_t handle, unsigned int flags)


.. py:function:: hipIpcCloseMemHandle(devPtr)

   Close memory mapped with hipIpcOpenMemHandle

   Unmaps memory returnd by hipIpcOpenMemHandle. The original allocation
   in the exporting process as well as imported mappings in other processes
   will be unaffected.

   Any resources used to enable peer access will be freed if this is the
   last mapping using them.

   Note:
       This IPC memory related feature API on Windows may behave differently from Linux.

   Args:
       devPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           - Device pointer returned by hipIpcOpenMemHandle

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorMapFailed`, :py:obj:`~.hipErrorInvalidHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipIpcCloseMemHandle(void * devPtr)


.. py:function:: hipIpcGetEventHandle(event)

   Gets an opaque interprocess handle for an event.

   This opaque handle may be copied into other processes and opened with hipIpcOpenEventHandle.
   Then hipEventRecord, hipEventSynchronize, hipStreamWaitEvent and hipEventQuery may be used in
   either process. Operations on the imported event after the exported event has been freed with
   hipEventDestroy will result in undefined behavior.

   Note:
       This IPC event related feature API is currently applicable on Linux.

   Args:
       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           Event allocated with hipEventInterprocess and hipEventDisableTiming flags

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidConfiguration`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipIpcEventHandle_st`:
               Pointer to hipIpcEventHandle to return the opaque event handle

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t * handle, hipEvent_t event)


.. py:function:: hipIpcOpenEventHandle(handle)

   Opens an interprocess event handles.

   Opens an interprocess event handle exported from another process with hipIpcGetEventHandle. The
   returned hipEvent_t behaves like a locally created event with the hipEventDisableTiming flag
   specified. This event need be freed with hipEventDestroy. Operations on the imported event after
   the exported event has been freed with hipEventDestroy will result in undefined behavior. If the
   function is called within the same process where handle is returned by hipIpcGetEventHandle, it
   will return hipErrorInvalidContext.

   Note:
       This IPC event related feature API is currently applicable on Linux.

   Args:
       handle (:py:obj:`~.hipIpcEventHandle_st`) -- *IN*:
           The opaque interprocess handle to open

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidContext`
       * :py:obj:`~.ihipEvent_t`:
               Pointer to hipEvent_t to return the event

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipIpcOpenEventHandle(hipEvent_t * event, hipIpcEventHandle_t handle)


.. py:function:: hipFuncSetAttribute(func, attr, value)

   Set attribute for a specific function

   *  This section describes the execution control functions of HIP runtime API.

   Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
   ignored on those architectures.

   Args:
       func (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer of the function

       attr (:py:obj:`~.hipFuncAttribute`) -- *IN*:
           Attribute to set

       value (:py:obj:`~.int`) -- *IN*:
           Value to set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDeviceFunction`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFuncSetAttribute(const void * func, hipFuncAttribute attr, int value)


.. py:function:: hipKernelSetAttribute(attrib, value, kernel, dev)

   Set attribute for a specific kernel

   Args:
       attrib (:py:obj:`~.hipFunction_attribute`) -- *IN*:
           Attribute to set

       value (:py:obj:`~.int`) -- *IN*:
           Value to set

       kernel (:py:obj:`~.ihipKernel_t`/:py:obj:`~.object`) -- *IN*:
           Kernel to set attribute for

       dev (:py:obj:`~.int`) -- *IN*:
           Device kernel execute on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidHandle`,
           :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidDeviceFunction`, :py:obj:`~.hipErrorMissingConfiguration`
           Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
           on those architectures.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipKernelSetAttribute(hipFunction_attribute attrib, int value, hipKernel_t kernel, hipDevice_t dev)


.. py:function:: hipKernelGetFunction(kernel)

   Function will be extracted for specific kernel

   Args:
       kernel (:py:obj:`~.ihipKernel_t`/:py:obj:`~.object`) -- *IN*:
           kernel to get handle for

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotFound`
       * :py:obj:`~.ihipModuleSymbol_t`:
               Pointer to function handle for the kernel

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipKernelGetFunction(hipFunction_t * pFunc, hipKernel_t kernel)


.. py:function:: hipFuncSetCacheConfig(func, config)

   Set Cache configuration for a specific function

   Args:
       func (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer of the function.

       config (:py:obj:`~.hipFuncCache_t`) -- *IN*:
           Configuration to set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`
           Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
           on those architectures.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFuncSetCacheConfig(const void * func, hipFuncCache_t config)


.. py:function:: hipFuncSetSharedMemConfig(func, config)

   Set shared memory configuation for a specific function

   Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
   ignored on those architectures.

   Args:
       func (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer of the function

       config (:py:obj:`~.hipSharedMemConfig`) -- *IN*:
           Configuration

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDeviceFunction`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFuncSetSharedMemConfig(const void * func, hipSharedMemConfig config)


.. py:function:: hipGetLastError()

   Return last error returned by any HIP runtime API call and resets the stored error code to
   :py:obj:`~.hipSuccess`

   *  This section describes the error handling functions of HIP runtime API.

   Returns the last error that has been returned by any of the runtime calls in the same host
   thread, and then resets the saved error to :py:obj:`~.hipSuccess`.

   See:
       :py:obj:`~.hipGetErrorString`, :py:obj:`~.hipGetLastError`, :py:obj:`~.hipPeakAtLastError`, :py:obj:`~.hipError_t`

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: return code from last HIP called from the active host thread

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetLastError()


.. py:function:: hipExtGetLastError()

   Return last error returned by any HIP runtime API call and resets the stored error code to
   :py:obj:`~.hipSuccess`

   Returns the last error that has been returned by any of the runtime calls in the same host
   thread, and then resets the saved error to :py:obj:`~.hipSuccess`.

   See:
       :py:obj:`~.hipGetErrorString`, :py:obj:`~.hipGetLastError`, :py:obj:`~.hipPeakAtLastError`, :py:obj:`~.hipError_t`

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: return code from last HIP called from the active host thread

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtGetLastError()


.. py:function:: hipPeekAtLastError()

   Return last error returned by any HIP runtime API call.

   Returns the last error that has been returned by any of the runtime calls in the same host
   thread. Unlike hipGetLastError, this function does not reset the saved error code.

   See:
       :py:obj:`~.hipGetErrorString`, :py:obj:`~.hipGetLastError`, :py:obj:`~.hipPeakAtLastError`, :py:obj:`~.hipError_t`

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipPeekAtLastError()


.. py:function:: hipGetErrorName(hip_error)

   Return hip error as text string form.

   See:
       :py:obj:`~.hipGetErrorString`, :py:obj:`~.hipGetLastError`, :py:obj:`~.hipPeakAtLastError`, :py:obj:`~.hipError_t`

   Args:
       hip_error (:py:obj:`~.hipError_t`):
           Error code to convert to name.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`:
               Always returns `~.hipError_t.hipSuccess`.
       * :py:obj:`~.bytes`: const char pointer to the NULL-terminated error name

   .. rubric:: C signature

   .. code-block:: c

       const char * hipGetErrorName(hipError_t hip_error)


.. py:function:: hipGetErrorString(hipError)

   Return handy text string message to explain the error which occurred

   See:
       :py:obj:`~.hipGetErrorName`, :py:obj:`~.hipGetLastError`, :py:obj:`~.hipPeakAtLastError`, :py:obj:`~.hipError_t`

   Args:
       hipError (:py:obj:`~.hipError_t`):
           Error code to convert to string.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`:
               Always returns `~.hipError_t.hipSuccess`.
       * :py:obj:`~.bytes`: const char pointer to the NULL-terminated error string

   .. rubric:: C signature

   .. code-block:: c

       const char * hipGetErrorString(hipError_t hipError)


.. py:function:: hipDrvGetErrorName(hipError)

   Return hip error as text string form.

   See:
       :py:obj:`~.hipGetErrorName`, :py:obj:`~.hipGetLastError`, :py:obj:`~.hipPeakAtLastError`, :py:obj:`~.hipError_t`

   Args:
       hipError (:py:obj:`~.hipError_t`) -- *IN*:
           Error code to convert to string.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               char pointer to the NULL-terminated error string

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvGetErrorName(hipError_t hipError, const char ** errorString)


.. py:function:: hipDrvGetErrorString(hipError)

   Return handy text string message to explain the error which occurred

   See:
       :py:obj:`~.hipGetErrorName`, :py:obj:`~.hipGetLastError`, :py:obj:`~.hipPeakAtLastError`, :py:obj:`~.hipError_t`

   Args:
       hipError (:py:obj:`~.hipError_t`) -- *IN*:
           Error code to convert to string.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               char pointer to the NULL-terminated error string

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvGetErrorString(hipError_t hipError, const char ** errorString)


.. py:function:: hipStreamCreate()

   Creates an asynchronous stream.

   Creates a new asynchronous stream with its associated current device. The ``stream`` returns an
   opaque handle that can be used to reference the newly created stream in subsequent hipStream*
   commands. The stream is allocated on the heap and will remain allocated even if the handle goes
   out-of-scope. To release the memory used by the stream, the application must call
   hipStreamDestroy.

   See:
       :py:obj:`~.hipStreamCreateWithFlags`, :py:obj:`~.hipStreamCreateWithPriority`, :py:obj:`~.hipStreamSynchronize`,
       :py:obj:`~.hipStreamWaitEvent`, :py:obj:`~.hipStreamDestroy`

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: One of:
               - py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
               - py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipStream_t`:
               Valid pointer to hipStream_t.  This function writes the memory with the
               newly created stream.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamCreate(hipStream_t * stream)


.. py:function:: hipStreamCreateWithFlags(flags)

   Creates an asynchronous stream with flag.

   Creates a new asynchronous stream with its associated current device. ``stream`` returns an
   opaque handle that can be used to reference the newly created stream in subsequent hipStream*
   commands. The stream is allocated on the heap and will remain allocated even if the handle
   goes out-of-scope. To release the memory used by the stream, application must call
   hipStreamDestroy.

   The ``flags`` parameter controls behavior of the stream. The valid values are :py:obj:`~.hipStreamDefault`
   and :py:obj:`~.hipStreamNonBlocking`.

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamCreateWithPriority`, :py:obj:`~.hipStreamSynchronize`, :py:obj:`~.hipStreamWaitEvent`,
       :py:obj:`~.hipStreamDestroy`.

   Args:
       flags (:py:obj:`~.int`) -- *IN*:
           Parameters to control stream creation

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipStream_t`:
               Pointer to new stream

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamCreateWithFlags(hipStream_t * stream, unsigned int flags)


.. py:function:: hipStreamCreateWithPriority(flags, priority)

   Creates an asynchronous stream with the specified priority.

   Creates a new asynchronous stream with the specified priority, with its associated current
   device.
   ``stream`` returns an opaque handle that can be used to reference the newly created stream in
   subsequent hipStream* commands. The stream is allocated on the heap and will remain allocated
   even if the handle goes out-of-scope. To release the memory used by the stream, application must
   call hipStreamDestroy.

   The ``flags`` parameter controls behavior of the stream. The valid values are :py:obj:`~.hipStreamDefault`
   and :py:obj:`~.hipStreamNonBlocking`.

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamSynchronize`, :py:obj:`~.hipStreamWaitEvent`, :py:obj:`~.hipStreamDestroy`

   Args:
       flags (:py:obj:`~.int`) -- *IN*:
           Parameters to control stream creation

       priority (:py:obj:`~.int`) -- *IN*:
           Priority of the stream. Lower numbers represent higher priorities.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipStream_t`:
               Pointer to new stream

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamCreateWithPriority(hipStream_t * stream, unsigned int flags, int priority)


.. py:function:: hipDeviceGetStreamPriorityRange()

   Returns numerical values that correspond to the least and greatest stream priority.

   Returns in *leastPriority and *greatestPriority the numerical values that correspond to the
   least and greatest stream priority respectively. Stream priorities follow a convention where
   lower numbers imply greater priorities. The range of meaningful stream priorities is given by
   [*leastPriority,*greatestPriority]. If the user attempts to create a stream with a priority
   value that is outside the meaningful range as specified by this API, the priority is
   automatically clamped to within the valid range.

   Warning:
       This API is under development on AMD GPUs and simply returns :py:obj:`~.hipSuccess`.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`
       * :py:obj:`~.int`:
               Pointer in which a value corresponding to least priority
               is returned.
       * :py:obj:`~.int`:
               Pointer in which a value corresponding to greatest priority
               is returned.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority)


.. py:function:: hipStreamDestroy(stream)

   Destroys the specified stream.

   Destroys the specified stream.

   If commands are still executing on the specified stream, some may complete execution before the
   queue is deleted.

   The queue may be destroyed while some commands are still inflight, or may wait for all commands
   queued to the stream before destroying it.

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamCreateWithFlags`, :py:obj:`~.hipStreamCreateWithPriority`, :py:obj:`~.hipStreamQuery`,
       :py:obj:`~.hipStreamWaitEvent`, :py:obj:`~.hipStreamSynchronize`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess` :py:obj:`~.hipErrorInvalidHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamDestroy(hipStream_t stream)


.. py:function:: hipStreamQuery(stream)

   Returns :py:obj:`~.hipSuccess` if all of the operations in the specified ``stream`` have completed, or
   :py:obj:`~.hipErrorNotReady` if not.

   This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
   host threads are sending work to the stream, the status may change immediately after the function
   is called.  It is typically used for debug.

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamCreateWithFlags`, :py:obj:`~.hipStreamCreateWithPriority`, :py:obj:`~.hipStreamWaitEvent`,
       :py:obj:`~.hipStreamSynchronize`, :py:obj:`~.hipStreamDestroy`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to query

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotReady`, :py:obj:`~.hipErrorInvalidHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamQuery(hipStream_t stream)


.. py:function:: hipStreamSynchronize(stream)

   Waits for all commands in the stream to complete.

   This command is host-synchronous : the host will block until all operations on the specified
   stream with its associated device are completed. On multiple device systems, the ``stream`` is
   associated with its device, no need to call hipSetDevice before this API.

   This command follows standard null-stream semantics. Specifying the null stream will cause the
   command to wait for other streams on the same device to complete all pending operations.

   This command honors the :py:obj:`~.hipDeviceScheduleBlockingSync` flag, which controls whether the wait is
   active or blocking.

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamCreateWithFlags`, :py:obj:`~.hipStreamCreateWithPriority`, :py:obj:`~.hipStreamWaitEvent`,
       :py:obj:`~.hipStreamDestroy`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamSynchronize(hipStream_t stream)


.. py:function:: hipStreamWaitEvent(stream, event, flags)

   Makes the specified compute stream wait for the specified event

   This function inserts a wait operation into the specified stream.
   All future work submitted to ``stream`` will wait until ``event`` reports completion before
   beginning execution.

   Flags include:
     hipEventWaitDefault: Default event creation flag.
     hipEventWaitExternal: Wait is captured in the graph as an external event node when
                             performing stream capture

   This function only waits for commands in the current stream to complete.  Notably, this function
   does not implicitly wait for commands in the default stream to complete, even if the specified
   stream is created with hipStreamNonBlocking = 0.

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamCreateWithFlags`, :py:obj:`~.hipStreamCreateWithPriority`,
       :py:obj:`~.hipStreamSynchronize`, :py:obj:`~.hipStreamDestroy`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to make wait

       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           Event to wait on

       flags (:py:obj:`~.int`) -- *IN*:
           Parameters to control the operation

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorStreamCaptureIsolation`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)


.. py:function:: hipStreamGetFlags(stream)

   Returns flags associated with this stream.

   See:
       :py:obj:`~.hipStreamCreateWithFlags`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to be queried

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidHandle`.
       * :py:obj:`~.int`:
               Pointer to an unsigned integer in which the stream's flags are returned

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int * flags)


.. py:function:: hipStreamGetId(stream)

   Queries the Id of a stream.

   See:
       :py:obj:`~.hipStreamCreateWithFlags`, :py:obj:`~.hipStreamGetFlags`, :py:obj:`~.hipStreamCreateWithPriority`, :py:obj:`~.hipStreamGetPriority`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to be queried

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidHandle`.
       * streamId (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetId(hipStream_t stream, unsigned long long * streamId)


.. py:function:: hipStreamGetPriority(stream)

   Queries the priority of a stream.

   See:
       :py:obj:`~.hipStreamCreateWithPriority`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to be queried

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidHandle`.
       * :py:obj:`~.int`:
               Pointer to an unsigned integer in which the stream's priority is
               returned

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetPriority(hipStream_t stream, int * priority)


.. py:function:: hipStreamGetDevice(stream)

   Gets the device associated with the stream.

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamDestroy`, :py:obj:`~.hipDeviceGetStreamPriorityRange`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to be queried

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorContextIsDestroyed`, :py:obj:`~.hipErrorInvalidHandle`,
           :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorInvalidContext`
       * :py:obj:`~.int`:
               Device associated with the stream

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t * device)


.. py:function:: hipExtStreamCreateWithCUMask(cuMaskSize, cuMask)

   Creates an asynchronous stream with the specified CU mask.

   Creates  a new asynchronous stream with the specified CU mask.  ``stream`` returns an opaque
   handle that can be used to reference the newly created stream in subsequent hipStream* commands.
   The stream is allocated on the heap and will remain allocated even if the handle goes
   out-of-scope. To release the memory used by the stream, application must call hipStreamDestroy.

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamSynchronize`, :py:obj:`~.hipStreamWaitEvent`, :py:obj:`~.hipStreamDestroy`

   Args:
       cuMaskSize (:py:obj:`~.int`) -- *IN*:
           Size of CU mask bit array passed in.

       cuMask (:py:obj:`~.rocm.bindings.util.types.ListOfUnsigned`/:py:obj:`~.object`) -- *IN*:
           Bit-vector representing the CU mask. Each active bit represents using one CU.
           The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
           CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are ignored.
           It is user's responsibility to make sure the input is meaningful.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipStream_t`:
               Pointer to new stream

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtStreamCreateWithCUMask(hipStream_t * stream, uint32_t cuMaskSize, const uint32_t * cuMask)


.. py:function:: hipExtStreamGetCUMask(stream, cuMaskSize, cuMask)

   Gets CU mask associated with an asynchronous stream

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamSynchronize`, :py:obj:`~.hipStreamWaitEvent`, :py:obj:`~.hipStreamDestroy`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to be queried

       cuMaskSize (:py:obj:`~.int`) -- *IN*:
           Number of the block of memories (uint32_t *) allocated by user

       cuMask (:py:obj:`~.rocm.bindings.util.types.ListOfUnsigned`/:py:obj:`~.object`) -- *OUT*:
           Pointer to a pre-allocated block of memories (uint32_t *) in which
           the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
           each active bit represents one active CU.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t * cuMask)


.. py:class:: hipStreamCallback_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Stream CallBack struct
       


.. py:function:: hipStreamAddCallback(stream, callback, userData, flags)

   Adds a callback to be called on the host after all currently enqueued items in the stream
   have completed.  For each hipStreamAddCallback call, a callback will be executed exactly once.
   The callback will block later work in the stream until it is finished.

   See:
       :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamCreateWithFlags`, :py:obj:`~.hipStreamQuery`, :py:obj:`~.hipStreamSynchronize`,
       :py:obj:`~.hipStreamWaitEvent`, :py:obj:`~.hipStreamDestroy`, :py:obj:`~.hipStreamCreateWithPriority`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream to add callback to

       callback (:py:obj:`~.hipStreamCallback_t`/:py:obj:`~.object`) -- *IN*:
           - The function to call once preceding stream operations are complete

       userData (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - User specified data to be passed to the callback function

       flags (:py:obj:`~.int`) -- *IN*:
           - Reserved for future use, must be 0

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void * userData, unsigned int flags)


.. py:function:: hipStreamSetAttribute(stream, attr, value)

   Sets stream attribute. Updated attribute is applied to work submitted to the stream.

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream to set attributes to

       attr (:py:obj:`~.hipLaunchAttributeID`) -- *IN*:
           - Attribute ID for the attribute to set

       value (:py:obj:`~.hipLaunchAttributeValue`/:py:obj:`~.object`) -- *IN*:
           - Attribute value for the attribute to set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamSetAttribute(hipStream_t stream, hipLaunchAttributeID attr, const hipLaunchAttributeValue * value)


.. py:function:: hipStreamGetAttribute(stream, attr, value_out)

   queries stream attribute.

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream to geet attributes from

       attr (:py:obj:`~.hipLaunchAttributeID`) -- *IN*:
           - Attribute ID for the attribute to query

       value_out (:py:obj:`~.hipLaunchAttributeValue`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetAttribute(hipStream_t stream, hipLaunchAttributeID attr, hipLaunchAttributeValue * value_out)


.. py:function:: hipStreamCopyAttributes(dst, src)

   Copies attributes from source stream to destination stream.

   Args:
       dst (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Destination stream

       src (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Source stream

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamCopyAttributes(hipStream_t dst, hipStream_t src)


.. py:function:: hipStreamWaitValue32(stream, ptr, value, flags, mask)

   Enqueues a wait command to the stream.[BETA]

   Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
   not execute until the defined wait condition is true.

   :py:obj:`~.hipStreamWaitValueGte`: waits until *ptr&mask >= value

   :py:obj:`~.hipStreamWaitValueEq` : waits until *ptr&mask == value

   :py:obj:`~.hipStreamWaitValueAnd`: waits until ((*ptr&mask) & value) != 0

   :py:obj:`~.hipStreamWaitValueNor`: waits until ~((*ptr&mask) | (value&mask)) != 0

   Note:
       when using :py:obj:`~.hipStreamWaitValueNor`, mask is applied on both 'value' and '*ptr'.

   Note:
       Support for :py:obj:`~.hipStreamWaitValue32` can be queried using 'hipDeviceGetAttribute()' and
       'hipDeviceAttributeCanUseStreamWaitValue' flag.

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   See:
       :py:obj:`~.hipExtMallocWithFlags`, :py:obj:`~.hipFree`, :py:obj:`~.hipStreamWaitValue64`, :py:obj:`~.hipStreamWriteValue64`,
       :py:obj:`~.hipStreamWriteValue32`, :py:obj:`~.hipDeviceGetAttribute`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream identifier

       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory object allocated using :py:obj:`~.hipMallocSignalMemory` flag

       value (:py:obj:`~.int`) -- *IN*:
           - Value to be used in compare operation

       flags (:py:obj:`~.int`) -- *IN*:
           - Defines the compare operation, supported values are :py:obj:`~.hipStreamWaitValueGte`
           :py:obj:`~.hipStreamWaitValueEq`, :py:obj:`~.hipStreamWaitValueAnd` and :py:obj:`~.hipStreamWaitValueNor`

       mask (:py:obj:`~.int`) -- *IN*:
           - Mask to be applied on value at memory before it is compared with value,
           default value is set to enable every bit

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamWaitValue32(hipStream_t stream, void * ptr, uint32_t value, unsigned int flags, uint32_t mask)


.. py:function:: hipStreamWaitValue64(stream, ptr, value, flags, mask)

   Enqueues a wait command to the stream.[BETA]

   Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
   not execute until the defined wait condition is true.

   :py:obj:`~.hipStreamWaitValueGte`: waits until *ptr&mask >= value

   :py:obj:`~.hipStreamWaitValueEq` : waits until *ptr&mask == value

   :py:obj:`~.hipStreamWaitValueAnd`: waits until ((*ptr&mask) & value) != 0

   :py:obj:`~.hipStreamWaitValueNor`: waits until ~((*ptr&mask) | (value&mask)) != 0

   Note:
       when using :py:obj:`~.hipStreamWaitValueNor`, mask is applied on both 'value' and '*ptr'.

   Note:
       Support for hipStreamWaitValue64 can be queried using 'hipDeviceGetAttribute()' and
       'hipDeviceAttributeCanUseStreamWaitValue' flag.

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   See:
       :py:obj:`~.hipExtMallocWithFlags`, :py:obj:`~.hipFree`, :py:obj:`~.hipStreamWaitValue32`, :py:obj:`~.hipStreamWriteValue64`,
       :py:obj:`~.hipStreamWriteValue32`, :py:obj:`~.hipDeviceGetAttribute`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream identifier

       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory object allocated using 'hipMallocSignalMemory' flag

       value (:py:obj:`~.int`) -- *IN*:
           - Value to be used in compare operation

       flags (:py:obj:`~.int`) -- *IN*:
           - Defines the compare operation, supported values are :py:obj:`~.hipStreamWaitValueGte`
           :py:obj:`~.hipStreamWaitValueEq`, :py:obj:`~.hipStreamWaitValueAnd` and :py:obj:`~.hipStreamWaitValueNor`.

       mask (:py:obj:`~.int`) -- *IN*:
           - Mask to be applied on value at memory before it is compared with value
           default value is set to enable every bit

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamWaitValue64(hipStream_t stream, void * ptr, uint64_t value, unsigned int flags, uint64_t mask)


.. py:function:: hipStreamWriteValue32(stream, ptr, value, flags)

   Enqueues a write command to the stream.[BETA]

   Enqueues a write command to the stream, write operation is performed after all earlier commands
   on this stream have completed the execution.

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   See:
       :py:obj:`~.hipExtMallocWithFlags`, :py:obj:`~.hipFree`, :py:obj:`~.hipStreamWriteValue32`, :py:obj:`~.hipStreamWaitValue32`,
       :py:obj:`~.hipStreamWaitValue64`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream identifier

       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to a GPU accessible memory object

       value (:py:obj:`~.int`) -- *IN*:
           - Value to be written

       flags (:py:obj:`~.int`) -- *IN*:
           - reserved, ignored for now, will be used in future releases

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamWriteValue32(hipStream_t stream, void * ptr, uint32_t value, unsigned int flags)


.. py:function:: hipStreamWriteValue64(stream, ptr, value, flags)

   Enqueues a write command to the stream.[BETA]

   Enqueues a write command to the stream, write operation is performed after all earlier commands
   on this stream have completed the execution.

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   See:
       :py:obj:`~.hipExtMallocWithFlags`, :py:obj:`~.hipFree`, :py:obj:`~.hipStreamWriteValue32`, :py:obj:`~.hipStreamWaitValue32`,
       :py:obj:`~.hipStreamWaitValue64`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream identifier

       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to a GPU accessible memory object

       value (:py:obj:`~.int`) -- *IN*:
           - Value to be written

       flags (:py:obj:`~.int`) -- *IN*:
           - reserved, ignored for now, will be used in future releases

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamWriteValue64(hipStream_t stream, void * ptr, uint64_t value, unsigned int flags)


.. py:function:: hipStreamBatchMemOp(stream, count, paramArray, flags)

   Enqueues an array of stream memory operations in the stream.[BETA]

   Batch operations to synchronize the stream via memory operations.

   Warning:
       This API is marked as beta, meaning, while this is feature complete,
       it is still open to changes and may have outstanding issues.

   See:
       :py:obj:`~.hipStreamWriteValue32`, :py:obj:`~.hipStreamWaitValue32`,
       :py:obj:`~.hipStreamWaitValue64`. :py:obj:`~.hipStreamWriteValue64`

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream identifier

       count (:py:obj:`~.int`) -- *IN*:
           - The number of operations in the array. Must be less than 256

       paramArray (:py:obj:`~.hipStreamBatchMemOpParams_union`/:py:obj:`~.object`) -- *IN*:
           - The types and parameters of the individual operations.

       flags (:py:obj:`~.int`) -- *IN*:
           - Reserved for future expansion; must be 0.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count, hipStreamBatchMemOpParams * paramArray, unsigned int flags)


.. py:function:: hipGraphAddBatchMemOpNode(hGraph, dependencies, numDependencies, nodeParams)

   Creates a batch memory operation node and adds it to a graph.[BETA]

   Warning:
       This API is marked as beta, meaning, while this is feature complete,
       it is still open to changes and may have outstanding issues.

   See:
       :py:obj:`~.hipStreamWriteValue32`, :py:obj:`~.hipStreamWaitValue32`,
       :py:obj:`~.hipStreamWaitValue64`. :py:obj:`~.hipStreamWriteValue64`, :py:obj:`~.hipStreamBatchMemOp`

   Args:
       hGraph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Graph to which to add the node

       dependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           -  Dependencies of the node

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies

       nodeParams (:py:obj:`~.hipBatchMemOpNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Parameters for the node

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Returns the newly created node

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddBatchMemOpNode(hipGraphNode_t * phGraphNode, hipGraph_t hGraph, const hipGraphNode_t * dependencies, size_t numDependencies, const hipBatchMemOpNodeParams * nodeParams)


.. py:function:: hipGraphBatchMemOpNodeGetParams(hNode, nodeParams_out)

   Returns a batch mem op node's parameters.[BETA]

   Returns the parameters of batch mem op node hNode in nodeParams_out.
   The paramArray returned in nodeParams_out is owned by the node.
   This memory remains valid until the node is destroyed or its parameters are modified,
   and should not be modified directly.

   Warning:
       This API is marked as beta, meaning, while this is feature complete,
       it is still open to changes and may have outstanding issues.

   See:
       :py:obj:`~.hipStreamWriteValue32`, :py:obj:`~.hipStreamWaitValue32`,
       :py:obj:`~.hipStreamWaitValue64`. :py:obj:`~.hipStreamWriteValue64`. :py:obj:`~.hipGraphBatchMemOpNodeSetParams`

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node to get the parameters for

       nodeParams_out (:py:obj:`~.hipBatchMemOpNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to return the parameters

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphBatchMemOpNodeGetParams(hipGraphNode_t hNode, hipBatchMemOpNodeParams * nodeParams_out)


.. py:function:: hipGraphBatchMemOpNodeSetParams(hNode, nodeParams)

   Sets the batch mem op node's parameters.[BETA]

   Sets the parameters of batch mem op node hNode to nodeParams.

   Warning:
       This API is marked as beta, meaning, while this is feature complete,
       it is still open to changes and may have outstanding issues.

   See:
       :py:obj:`~.hipStreamWriteValue32`, :py:obj:`~.hipStreamWaitValue32`,
       :py:obj:`~.hipStreamWaitValue64`. :py:obj:`~.hipStreamWriteValue64`, :py:obj:`~.hipGraphBatchMemOpNodeGetParams`

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node to set the parameters for

       nodeParams (:py:obj:`~.hipBatchMemOpNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Parameters to copy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphBatchMemOpNodeSetParams(hipGraphNode_t hNode, hipBatchMemOpNodeParams * nodeParams)


.. py:function:: hipGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams)

   Sets the parameters for a batch mem op node in the given graphExec.[BETA]

   Sets the parameters of a batch mem op node in an executable graph hGraphExec.
   The node is identified by the corresponding node hNode in the non-executable graph,
   from which the executable graph was instantiated.

   Warning:
       This API is marked as beta, meaning, while this is feature complete,
       it is still open to changes and may have outstanding issues.

   See:
       :py:obj:`~.hipStreamWriteValue32`, :py:obj:`~.hipStreamWaitValue32`,
       :py:obj:`~.hipStreamWaitValue64`. :py:obj:`~.hipStreamWriteValue64`, :py:obj:`~.hipStreamBatchMemOp`

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - The executable graph in which to set the specified node

       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Batch mem op node from the graph from which graphExec was instantiated

       nodeParams (:py:obj:`~.hipBatchMemOpNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Updated Parameters to set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipBatchMemOpNodeParams * nodeParams)


.. py:function:: hipEventCreateWithFlags(flags)

   Create an event with the specified flags

   *

   See:
       :py:obj:`~.hipEventCreate`, :py:obj:`~.hipEventSynchronize`, :py:obj:`~.hipEventDestroy`, :py:obj:`~.hipEventElapsedTime`

   Args:
       flags (:py:obj:`~.int`) -- *IN*:
           Flags to control event behavior.  Valid values are :py:obj:`~.hipEventDefault`,
           :py:obj:`~.hipEventBlockingSync`, :py:obj:`~.hipEventDisableTiming`, :py:obj:`~.hipEventInterprocess`
             :py:obj:`~.hipEventDefault` : Default flag.  The event will use active synchronization and will support
           timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a
           CPU to poll on the event.
             :py:obj:`~.hipEventBlockingSync` : The event will use blocking synchronization : if hipEventSynchronize is
           called on this event, the thread will block until the event completes.  This can increase latency
           for the synchroniation but can result in lower power and more resources for other CPU threads.
             :py:obj:`~.hipEventDisableTiming` : Disable recording of timing information. Events created with this flag
           would not record profiling data and provide best performance if used for synchronization.
             :py:obj:`~.hipEventInterprocess` : The event can be used as an interprocess event. hipEventDisableTiming
           flag also must be set when hipEventInterprocess flag is set.
             :py:obj:`~.hipEventDisableSystemFence` : Disable acquire and release system scope fence. This may
           improve performance but device memory may not be visible to the host and other devices
           if this flag is set.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorLaunchFailure`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.ihipEvent_t`:
               Returns the newly created event.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipEventCreateWithFlags(hipEvent_t * event, unsigned int flags)


.. py:function:: hipEventCreate()

   Create an event

   See:
       :py:obj:`~.hipEventCreateWithFlags`, :py:obj:`~.hipEventRecord`, :py:obj:`~.hipEventQuery`, :py:obj:`~.hipEventSynchronize`,
       :py:obj:`~.hipEventDestroy`, :py:obj:`~.hipEventElapsedTime`

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorLaunchFailure`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.ihipEvent_t`:
               Returns the newly created event.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipEventCreate(hipEvent_t * event)


.. py:function:: hipEventRecordWithFlags(event, stream, flags)

   Record an event in the specified stream.

   hipEventQuery() or hipEventSynchronize() must be used to determine when the event
   transitions from "recording" (after hipEventRecord() is called) to "recorded"
   (when timestamps are set, if requested).

   Events which are recorded in a non-NULL stream will transition to
   from recording to "recorded" state when they reach the head of
   the specified stream, after all previous
   commands in that stream have completed executing.

   Flags include:
     hipEventRecordDefault: Default event creation flag.
     hipEventRecordExternal: Event is captured in the graph as an external event node when
                             performing stream capture

   If hipEventRecord() has been previously called on this event, then this call will overwrite any
   existing state in event.

   If this function is called on an event that is currently being recorded, results are undefined
   - either outstanding recording may save state into the event, and the order is not guaranteed.

   Note:
       If this function is not called before use hipEventQuery() or hipEventSynchronize(),
       :py:obj:`~.hipSuccess` is returned, meaning no pending event in the stream.

   See:
       :py:obj:`~.hipEventCreate`, :py:obj:`~.hipEventCreateWithFlags`, :py:obj:`~.hipEventQuery`, :py:obj:`~.hipEventSynchronize`,
       :py:obj:`~.hipEventDestroy`, :py:obj:`~.hipEventElapsedTime`

   Args:
       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           event to record.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream in which to record event.

       flags (:py:obj:`~.int`) -- *IN*:
           parameter for operations

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`,
           :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorLaunchFailure`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned int flags)


.. py:function:: hipEventRecord(event, stream)

   Record an event in the specified stream.

   hipEventQuery() or hipEventSynchronize() must be used to determine when the event
   transitions from "recording" (after hipEventRecord() is called) to "recorded"
   (when timestamps are set, if requested).

   Events which are recorded in a non-NULL stream will transition to
   from recording to "recorded" state when they reach the head of
   the specified stream, after all previous
   commands in that stream have completed executing.

   If hipEventRecord() has been previously called on this event, then this call will overwrite any
   existing state in event.

   If this function is called on an event that is currently being recorded, results are undefined
   - either outstanding recording may save state into the event, and the order is not guaranteed.

   Note:
       If this function is not called before use hipEventQuery() or hipEventSynchronize(),
       :py:obj:`~.hipSuccess` is returned, meaning no pending event in the stream.

   See:
       :py:obj:`~.hipEventCreate`, :py:obj:`~.hipEventCreateWithFlags`, :py:obj:`~.hipEventQuery`, :py:obj:`~.hipEventSynchronize`,
       :py:obj:`~.hipEventDestroy`, :py:obj:`~.hipEventElapsedTime`

   Args:
       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           event to record.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream in which to record event.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`,
           :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorLaunchFailure`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)


.. py:function:: hipEventDestroy(event)

   Destroy the specified event.

   Releases memory associated with the event.  If the event is recording but has not completed
   recording when hipEventDestroy() is called, the function will return immediately and the
   completion_future resources will be released later, when the hipDevice is synchronized.

   See:
       :py:obj:`~.hipEventCreate`, :py:obj:`~.hipEventCreateWithFlags`, :py:obj:`~.hipEventQuery`, :py:obj:`~.hipEventSynchronize`, :py:obj:`~.hipEventRecord`,
       :py:obj:`~.hipEventElapsedTime`

   Args:
       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           Event to destroy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: One of:
               - py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`,
                     :py:obj:`~.hipErrorLaunchFailure`
               - py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipEventDestroy(hipEvent_t event)


.. py:function:: hipEventSynchronize(event)

   Wait for an event to complete.

   This function will block until the event is ready, waiting for all previous work in the stream
   specified when event was recorded with hipEventRecord().

    If hipEventRecord() has not been called on ``event,`` this function returns :py:obj:`~.hipSuccess` when no
    event is captured.

   See:
       :py:obj:`~.hipEventCreate`, :py:obj:`~.hipEventCreateWithFlags`, :py:obj:`~.hipEventQuery`, :py:obj:`~.hipEventDestroy`, :py:obj:`~.hipEventRecord`,
       :py:obj:`~.hipEventElapsedTime`

   Args:
       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           Event on which to wait.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`,
           :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorLaunchFailure`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipEventSynchronize(hipEvent_t event)


.. py:function:: hipEventElapsedTime(start, stop)

   Return the elapsed time between two events.

   Computes the elapsed time between two events. Time is computed in ms, with
   a resolution of approximately 1 us.

   Events which are recorded in a NULL stream will block until all commands
   on all other streams complete execution, and then record the timestamp.

   Events which are recorded in a non-NULL stream will record their timestamp
   when they reach the head of the specified stream, after all previous
   commands in that stream have completed executing.  Thus the time that
   the event recorded may be significantly after the host calls hipEventRecord().

   If hipEventRecord() has not been called on either event, then :py:obj:`~.hipErrorInvalidHandle` is
   returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
   recorded on one or both events (that is, hipEventQuery() would return :py:obj:`~.hipErrorNotReady` on at
   least one of the events), then :py:obj:`~.hipErrorNotReady` is returned.

   See:
       :py:obj:`~.hipEventCreate`, :py:obj:`~.hipEventCreateWithFlags`, :py:obj:`~.hipEventQuery`, :py:obj:`~.hipEventDestroy`, :py:obj:`~.hipEventRecord`,
       :py:obj:`~.hipEventSynchronize`

   Args:
       start (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           Start event.

       stop (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           Stop event.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotReady`, :py:obj:`~.hipErrorInvalidHandle`,
           :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorLaunchFailure`
       * :py:obj:`~.float`:
               Return time between start and stop in ms.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipEventElapsedTime(float * ms, hipEvent_t start, hipEvent_t stop)


.. py:function:: hipEventQuery(event)

   Query event status

   Query the status of the specified event.  This function will return :py:obj:`~.hipSuccess` if all
   commands in the appropriate stream (specified to hipEventRecord()) have completed.  If any
   execution has not completed, then :py:obj:`~.hipErrorNotReady` is returned.

   Note:
       This API returns :py:obj:`~.hipSuccess`, if hipEventRecord() is not called before this API.

   See:
       :py:obj:`~.hipEventCreate`, :py:obj:`~.hipEventCreateWithFlags`, :py:obj:`~.hipEventRecord`, :py:obj:`~.hipEventDestroy`,
       :py:obj:`~.hipEventSynchronize`, :py:obj:`~.hipEventElapsedTime`

   Args:
       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           Event to query.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotReady`, :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorLaunchFailure`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipEventQuery(hipEvent_t event)


.. py:function:: hipPointerSetAttribute(value, attribute, ptr)

   Sets information on the specified pointer.[BETA]

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Args:
       value (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Sets pointer attribute value

       attribute (:py:obj:`~.hipPointer_attribute`) -- *IN*:
           Attribute to set

       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to set attributes for

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipPointerSetAttribute(const void * value, hipPointer_attribute attribute, hipDeviceptr_t ptr)


.. py:function:: hipPointerGetAttributes(ptr)

   Returns attributes for the specified pointer

   The output parameter 'attributes' has a member named 'type' that describes what memory the
   pointer is associated with, such as device memory, host memory, managed memory, and others.
   Otherwise, the API cannot handle the pointer and returns :py:obj:`~.hipErrorInvalidValue`.

   Note:
       The unrecognized memory type is unsupported to keep the HIP functionality backward
       compatibility due to :py:obj:`~.hipMemoryType` enum values.

   Note:
       The current behavior of this HIP API corresponds to the CUDA API before version 11.0.

   See:
       :py:obj:`~.hipPointerGetAttribute`

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to get attributes for

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipPointerAttribute_t`:
               attributes for the specified pointer

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipPointerGetAttributes(hipPointerAttribute_t * attributes, const void * ptr)


.. py:function:: hipPointerGetAttribute(data, attribute, ptr)

   Returns information about the specified pointer.[BETA]

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   See:
       :py:obj:`~.hipPointerGetAttributes`

   Args:
       data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Returned pointer attribute value

       attribute (:py:obj:`~.hipPointer_attribute`) -- *IN*:
           Attribute to query for

       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to get attributes for

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipPointerGetAttribute(void * data, hipPointer_attribute attribute, hipDeviceptr_t ptr)


.. py:function:: hipDrvPointerGetAttributes(numAttributes, ptr)

   Returns information about the specified pointer.[BETA]

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   See:
       :py:obj:`~.hipPointerGetAttribute`

   Args:
       numAttributes (:py:obj:`~.int`) -- *IN*:
           number of attributes to query for

       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to get attributes for

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipPointer_attribute`:
               attributes to query for
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               a two-dimensional containing pointers to memory locations
               where the result of each attribute query will be written to

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute * attributes, void ** data, hipDeviceptr_t ptr)


.. py:function:: hipImportExternalSemaphore(semHandleDesc)

   Imports an external semaphore.

   *  

   This section describes the external resource interoperability functions of HIP runtime API.

   See:

   Note:
       This API is currently not supported on Linux.

   Args:
       semHandleDesc (:py:obj:`~.hipExternalSemaphoreHandleDesc_st`/:py:obj:`~.object`) -- *IN*:
           Semaphore import handle descriptor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               External semaphores to be waited on

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t * extSem_out, const hipExternalSemaphoreHandleDesc * semHandleDesc)


.. py:function:: hipSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

   Signals a set of external semaphore objects.

   See:

   Note:
       This API is currently not supported on Linux.

   Args:
       extSemArray (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           External semaphores to be waited on

       paramsArray (:py:obj:`~.hipExternalSemaphoreSignalParams_st`/:py:obj:`~.object`) -- *IN*:
           Array of semaphore parameters

       numExtSems (:py:obj:`~.int`) -- *IN*:
           Number of semaphores to wait on

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to enqueue the wait operations in

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t * extSemArray, const hipExternalSemaphoreSignalParams * paramsArray, unsigned int numExtSems, hipStream_t stream)


.. py:function:: hipWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

   Waits on a set of external semaphore objects

   See:

   Note:
       This API is currently not supported on Linux.

   Args:
       extSemArray (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           External semaphores to be waited on

       paramsArray (:py:obj:`~.hipExternalSemaphoreWaitParams_st`/:py:obj:`~.object`) -- *IN*:
           Array of semaphore parameters

       numExtSems (:py:obj:`~.int`) -- *IN*:
           Number of semaphores to wait on

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to enqueue the wait operations in

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t * extSemArray, const hipExternalSemaphoreWaitParams * paramsArray, unsigned int numExtSems, hipStream_t stream)


.. py:function:: hipDestroyExternalSemaphore(extSem)

   Destroys an external semaphore object and releases any references to the underlying
   resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.

   See:

   Note:
       This API is currently not supported on Linux.

   Args:
       extSem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           handle to an external memory object

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem)


.. py:function:: hipImportExternalMemory(memHandleDesc)

   Imports an external memory object.

   See:

   Args:
       memHandleDesc (:py:obj:`~.hipExternalMemoryHandleDesc_st`/:py:obj:`~.object`) -- *IN*:
           Memory import handle descriptor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Returned handle to an external memory object

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipImportExternalMemory(hipExternalMemory_t * extMem_out, const hipExternalMemoryHandleDesc * memHandleDesc)


.. py:function:: hipExternalMemoryGetMappedBuffer(extMem, bufferDesc)

   Maps a buffer onto an imported memory object.

   See:

   Args:
       extMem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Handle to external memory object

       bufferDesc (:py:obj:`~.hipExternalMemoryBufferDesc_st`/:py:obj:`~.object`) -- *IN*:
           Buffer descriptor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Returned device pointer to buffer

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExternalMemoryGetMappedBuffer(void ** devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc * bufferDesc)


.. py:function:: hipDestroyExternalMemory(extMem)

   Destroys an external memory object.

   See:

   Args:
       extMem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           External memory object to be destroyed

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem)


.. py:function:: hipExternalMemoryGetMappedMipmappedArray(extMem, mipmapDesc)

   Maps a mipmapped array onto an external memory object.

   Returned mipmapped array must be freed using hipFreeMipmappedArray.

   See:
       :py:obj:`~.hipImportExternalMemory`, :py:obj:`~.hipDestroyExternalMemory`, :py:obj:`~.hipExternalMemoryGetMappedBuffer`,
       :py:obj:`~.hipFreeMipmappedArray`

   Args:
       extMem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           external memory object handle

       mipmapDesc (:py:obj:`~.hipExternalMemoryMipmappedArrayDesc_st`/:py:obj:`~.object`) -- *IN*:
           external mipmapped array descriptor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceHandle`
       * :py:obj:`~.hipMipmappedArray`:
               mipmapped array to return

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExternalMemoryGetMappedMipmappedArray(hipMipmappedArray_t * mipmap, hipExternalMemory_t extMem, const hipExternalMemoryMipmappedArrayDesc * mipmapDesc)


.. py:function:: hipMalloc(size)

   Allocate memory on the default accelerator

   If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.

   See:
       :py:obj:`~.hipMallocPitch`, :py:obj:`~.hipFree`, :py:obj:`~.hipMallocArray`, :py:obj:`~.hipFreeArray`, :py:obj:`~.hipMalloc3D`, :py:obj:`~.hipMalloc3DArray`,
       :py:obj:`~.hipHostFree`, :py:obj:`~.hipHostMalloc`

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           Requested memory size

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`, :py:obj:`~.hipErrorInvalidValue` (bad context, null *ptr)
       * :py:obj:`~.rocm.bindings.util.types.DeviceArray`/:py:obj:`~.object`:
               Pointer to the allocated memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMalloc(void ** ptr, size_t size)


.. py:function:: hipExtMallocWithFlags(sizeBytes, flags)

   Allocate memory on the default accelerator

   If requested memory size is 0, no memory is allocated, *ptr returns nullptr, and :py:obj:`~.hipSuccess`
   is returned.

   The memory allocation flag should be either :py:obj:`~.hipDeviceMallocDefault`,
   :py:obj:`~.hipDeviceMallocFinegrained`, :py:obj:`~.hipDeviceMallocUncached`, or :py:obj:`~.hipMallocSignalMemory`.
   If the flag is any other value, the API returns :py:obj:`~.hipErrorInvalidValue`.

   See:
       :py:obj:`~.hipMallocPitch`, :py:obj:`~.hipFree`, :py:obj:`~.hipMallocArray`, :py:obj:`~.hipFreeArray`, :py:obj:`~.hipMalloc3D`, :py:obj:`~.hipMalloc3DArray`,
       :py:obj:`~.hipHostFree`, :py:obj:`~.hiHostMalloc`

   Args:
       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Requested memory size

       flags (:py:obj:`~.int`) -- *IN*:
           Type of memory allocation

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`, :py:obj:`~.hipErrorInvalidValue` (bad context, null *ptr)
       * :py:obj:`~.rocm.bindings.util.types.DeviceArray`/:py:obj:`~.object`:
               Pointer to the allocated memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtMallocWithFlags(void ** ptr, size_t sizeBytes, unsigned int flags)


.. py:function:: hipMallocHost(size)

   Allocate pinned host memory [Deprecated]

   If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.

   Warning:
       This API is deprecated, use hipHostMalloc() instead

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           Requested memory size

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to the allocated host pinned memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMallocHost(void ** ptr, size_t size)


.. py:function:: hipMemAllocHost(size)

   Allocate pinned host memory [Deprecated]

   If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.

   Warning:
       This API is deprecated, use hipHostMalloc() instead

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           Requested memory size

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to the allocated host pinned memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemAllocHost(void ** ptr, size_t size)


.. py:function:: hipHostMalloc(size, flags)

   Allocates device accessible page locked (pinned) host memory

   This API allocates pinned host memory which is mapped into the address space of all GPUs
   in the system, the memory can be accessed directly by the GPU device, and can be read or
   written with much higher bandwidth than pageable memory obtained with functions such as
   malloc().

   Using the pinned host memory, applications can implement faster data transfers for HostToDevice
   and DeviceToHost. The runtime tracks the hipHostMalloc allocations and can avoid some of the
   setup required for regular unpinned memory.

   When the memory accesses are infrequent, zero-copy memory can be a good choice, for coherent
   allocation. GPU can directly access the host memory over the CPU/GPU interconnect, without need
   to copy the data.

   Currently the allocation granularity is 4KB for the API.

   Developers need to choose proper allocation flag with consideration of synchronization.

   If no input for flags, it will be the default pinned memory allocation on the host.

   See:
       :py:obj:`~.hipSetDeviceFlags`, :py:obj:`~.hiptHostFree`

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           Requested memory size in bytes
           If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.

       flags (:py:obj:`~.int`) -- *IN*:
           Type of host memory allocation. See the description of flags in
           hipSetDeviceFlags.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to the allocated host pinned memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipHostMalloc(void ** ptr, size_t size, unsigned int flags)


.. py:function:: hipMallocManaged(size, flags)

   Allocates memory that will be automatically managed by HIP.

   *  This section describes the managed memory management functions of HIP runtime API.

   Note:
       The managed memory management APIs are implemented on Linux, under developement
       on Windows.

   This API is used for managed memory, allows data be shared and accessible to both CPU and
   GPU using a single pointer.

   The API returns the allocation pointer, managed by HMM, can be used further to execute kernels
   on device and fetch data between the host and device as needed.

   If HMM is not supported, the function behaves the same as ``hipMallocHost`` .

   Note:
       It is recommend to do the capability check before call this API.

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           - requested allocation size in bytes, it should be granularity of 4KB

       flags (:py:obj:`~.int`) -- *IN*:
           - must be either hipMemAttachGlobal or hipMemAttachHost
           (defaults to hipMemAttachGlobal)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorMemoryAllocation`, :py:obj:`~.hipErrorNotSupported`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.DeviceArray`/:py:obj:`~.object`:
               - pointer to allocated device memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMallocManaged(void ** dev_ptr, size_t size, unsigned int flags)


.. py:function:: hipMemPrefetchAsync(dev_ptr, count, device, stream)

   Prefetches memory to the specified destination device using HIP.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to be prefetched

       count (:py:obj:`~.int`) -- *IN*:
           size in bytes for prefetching

       device (:py:obj:`~.int`) -- *IN*:
           destination device to prefetch to

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream to enqueue prefetch operation

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPrefetchAsync(const void * dev_ptr, size_t count, int device, hipStream_t stream)


.. py:function:: hipMemPrefetchAsync_v2(dev_ptr, count, location, flags, stream)

   Prefetches memory to the specified destination device using HIP.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to be prefetched

       count (:py:obj:`~.int`) -- *IN*:
           size in bytes for prefetching

       location (:py:obj:`~.hipMemLocation`) -- *IN*:
           destination location to prefetch to

       flags (:py:obj:`~.int`) -- *IN*:
           flags for future use, must be zero now.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream to enqueue prefetch operation

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPrefetchAsync_v2(const void * dev_ptr, size_t count, hipMemLocation location, unsigned int flags, hipStream_t stream)


.. py:function:: hipMemPrefetchBatchAsync(dev_ptrs, count, prefetch_locs, num_prefetch_locs, flags, stream)

   Prefetches a batch of memory ranges to the specified locations using HIP.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       dev_ptrs (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           pointers to the memory ranges to prefetch

       count (:py:obj:`~.int`) -- *IN*:
           number of memory ranges to prefetch

       prefetch_locs (:py:obj:`~.hipMemLocation`/:py:obj:`~.object`) -- *IN*:
           locations to prefetch the memory ranges to

       num_prefetch_locs (:py:obj:`~.int`) -- *IN*:
           number of locations to prefetch

       flags (:py:obj:`~.int`) -- *IN*:
           flags for future use, must be zero now.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream to enqueue the prefetch operation

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               sizes in bytes of the memory ranges to prefetch
       * :py:obj:`~.int`:
               indices of the memory ranges to prefetch

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPrefetchBatchAsync(void ** dev_ptrs, size_t * sizes, size_t count, hipMemLocation * prefetch_locs, size_t * prefetch_loc_idxs, size_t num_prefetch_locs, unsigned long long flags, hipStream_t stream)


.. py:function:: hipMemDiscardBatchAsync(dev_ptrs, count, flags, stream)

   Discards a batch of memory ranges asynchronously.

   Warning:
       Reading from a discarded range without first writing or prefetching
       to it will return an indeterminate value.

   Warning:
       Concurrent reads, writes, or prefetches to discarded ranges result
       in undefined behavior.

   Note:
       All memory ranges must be managed memory allocated via hipMallocManaged
       or system-allocated memory (if device supports pageable memory access).

   Note:
       This API is implemented on Linux and requires XNACK to be enabled.

   Note:
       This API is marked as beta, meaning, while this is feature complete,
       it is still open to changes and may have outstanding issues.

   See:
       :py:obj:`~.hipMemPrefetchBatchAsync`, :py:obj:`~.hipMallocManaged`

   Args:
       dev_ptrs (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           pointers to the memory ranges to discard

       count (:py:obj:`~.int`) -- *IN*:
           number of memory ranges to discard

       flags (:py:obj:`~.int`) -- *IN*:
           flags for future use, must be zero now.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream to enqueue the discard operation

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               sizes in bytes of the memory ranges to discard

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemDiscardBatchAsync(void ** dev_ptrs, size_t * sizes, size_t count, unsigned long long flags, hipStream_t stream)


.. py:function:: hipDrvMemDiscardBatchAsync(dptrs, count, flags, stream)

   Discards a batch of memory ranges asynchronously (driver API variant).

   Warning:
       Reading from a discarded range without first writing or prefetching
       to it will return an indeterminate value.

   Note:
       This is the driver API variant that uses hipDeviceptr_t instead of void*.
       Both hipMemDiscardBatchAsync and hipDrvMemDiscardBatchAsync use the same
       internal implementation.

   See:
       :py:obj:`~.hipMemDiscardBatchAsync`, :py:obj:`~.hipMemPrefetchBatchAsync`, :py:obj:`~.hipMallocManaged`

   Args:
       dptrs (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           pointers to the memory ranges to discard

       count (:py:obj:`~.int`) -- *IN*:
           number of memory ranges to discard

       flags (:py:obj:`~.int`) -- *IN*:
           flags for future use, must be zero now.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream to enqueue the discard operation

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               sizes in bytes of the memory ranges to discard

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvMemDiscardBatchAsync(hipDeviceptr_t * dptrs, size_t * sizes, size_t count, unsigned long long flags, hipStream_t stream)


.. py:function:: hipMemDiscardAndPrefetchBatchAsync(dptrs, count, prefetchLocs, numPrefetchLocs, flags, stream)

   Discards and prefetches a batch of memory ranges asynchronously.

   Semantically equivalent to calling ``hipMemDiscardBatchAsync`` followed by
   ``hipMemPrefetchBatchAsync,`` but combines both operations into a single
   command submission for reduced overhead.

   Warning:
       Reading from a discarded range without first writing or prefetching
       to it will return an indeterminate value.

   Note:
       All memory ranges must be managed memory allocated via hipMallocManaged
       or system-allocated memory (if device supports pageable memory access).

   Note:
       This API is implemented on Linux and requires XNACK to be enabled.

   Note:
       This API is marked as beta, meaning, while this is feature complete,
       it is still open to changes and may have outstanding issues.

   See:
       :py:obj:`~.hipMemDiscardBatchAsync`, :py:obj:`~.hipMemPrefetchBatchAsync`, :py:obj:`~.hipMallocManaged`

   Args:
       dptrs (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           pointers to the memory ranges

       count (:py:obj:`~.int`) -- *IN*:
           number of memory ranges

       prefetchLocs (:py:obj:`~.hipMemLocation`/:py:obj:`~.object`) -- *IN*:
           array of target locations for prefetching

       numPrefetchLocs (:py:obj:`~.int`) -- *IN*:
           number of unique prefetch locations

       flags (:py:obj:`~.int`) -- *IN*:
           flags for future use, must be zero now.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream to enqueue the operation

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               sizes in bytes of the memory ranges
       * :py:obj:`~.int`:
               indices mapping each range to a prefetch location

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemDiscardAndPrefetchBatchAsync(void ** dptrs, size_t * sizes, size_t count, hipMemLocation * prefetchLocs, size_t * prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, hipStream_t stream)


.. py:function:: hipDrvMemDiscardAndPrefetchBatchAsync(dptrs, count, prefetchLocs, numPrefetchLocs, flags, stream)

   Discards and prefetches a batch of memory ranges asynchronously (driver API variant).

   Note:
       This is the driver API variant that uses hipDeviceptr_t instead of void*.

   See:
       :py:obj:`~.hipMemDiscardAndPrefetchBatchAsync`, :py:obj:`~.hipMemDiscardBatchAsync`, :py:obj:`~.hipMemPrefetchBatchAsync`

   Args:
       dptrs (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           pointers to the memory ranges

       count (:py:obj:`~.int`) -- *IN*:
           number of memory ranges

       prefetchLocs (:py:obj:`~.hipMemLocation`/:py:obj:`~.object`) -- *IN*:
           array of target locations for prefetching

       numPrefetchLocs (:py:obj:`~.int`) -- *IN*:
           number of unique prefetch locations

       flags (:py:obj:`~.int`) -- *IN*:
           flags for future use, must be zero now.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream to enqueue the operation

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               sizes in bytes of the memory ranges
       * :py:obj:`~.int`:
               indices mapping each range to a prefetch location

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvMemDiscardAndPrefetchBatchAsync(hipDeviceptr_t * dptrs, size_t * sizes, size_t count, hipMemLocation * prefetchLocs, size_t * prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, hipStream_t stream)


.. py:function:: hipMemAdvise(dev_ptr, count, advice, device)

   Advise about the usage of a given memory range to HIP.

   This HIP API advises about the usage to be applied on unified memory allocation in the
   range starting from the pointer address devPtr, with the size of count bytes.
   The memory range must refer to managed memory allocated via the API hipMallocManaged, and the
   range will be handled with proper round down and round up respectively in the driver to
   be aligned to CPU page size, the same way as corresponding CUDA API behaves in CUDA version 8.0
   and afterwards.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to memory to set the advice for

       count (:py:obj:`~.int`) -- *IN*:
           size in bytes of the memory range, it should be CPU page size alligned.

       advice (:py:obj:`~.hipMemoryAdvise`) -- *IN*:
           advice to be applied for the specified memory range

       device (:py:obj:`~.int`) -- *IN*:
           device to apply the advice for

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemAdvise(const void * dev_ptr, size_t count, hipMemoryAdvise advice, int device)


.. py:function:: hipMemAdvise_v2(dev_ptr, count, advice, location)

   Advise about the usage of a given memory range to HIP.

   This HIP API advises about the usage to be applied on unified memory allocation in the
   range starting from the pointer address devPtr, with the size of count bytes.
   The memory range must refer to managed memory allocated via the API hipMallocManaged, and the
   range will be handled with proper round down and round up respectively in the driver to
   be aligned to CPU page size, the same way as corresponding CUDA API behaves in CUDA version 8.0
   and afterwards.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to memory to set the advice for

       count (:py:obj:`~.int`) -- *IN*:
           size in bytes of the memory range, it should be CPU page size alligned.

       advice (:py:obj:`~.hipMemoryAdvise`) -- *IN*:
           advice to be applied for the specified memory range

       location (:py:obj:`~.hipMemLocation`) -- *IN*:
           location to apply the advice for

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemAdvise_v2(const void * dev_ptr, size_t count, hipMemoryAdvise advice, hipMemLocation location)


.. py:function:: hipMemRangeGetAttribute(data, data_size, attribute, dev_ptr, count)

   Query an attribute of a given memory range in HIP.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to a memory location where the result of each
           attribute query will be written to

       data_size (:py:obj:`~.int`) -- *IN*:
           the size of data

       attribute (:py:obj:`~.hipMemRangeAttribute`) -- *IN*:
           the attribute to query

       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           start of the range to query

       count (:py:obj:`~.int`) -- *IN*:
           size of the range to query

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemRangeGetAttribute(void * data, size_t data_size, hipMemRangeAttribute attribute, const void * dev_ptr, size_t count)


.. py:function:: hipMemRangeGetAttributes(num_attributes, dev_ptr, count)

   Query attributes of a given memory range in HIP.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       num_attributes (:py:obj:`~.int`) -- *IN*:
           an array of attributes to query (numAttributes and the number
           of attributes in this array should match)

       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           start of the range to query

       count (:py:obj:`~.int`) -- *IN*:
           size of the range to query

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               a two-dimensional array containing pointers to memory locations
               where the result of each attribute query will be written to
       * :py:obj:`~.int`:
               an array, containing the sizes of each result
       * :py:obj:`~.hipMemRangeAttribute`:
               the attribute to query

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemRangeGetAttributes(void ** data, size_t * data_sizes, hipMemRangeAttribute * attributes, size_t num_attributes, const void * dev_ptr, size_t count)


.. py:function:: hipStreamAttachMemAsync(stream, dev_ptr, length, flags)

   Attach memory to a stream asynchronously in HIP.

   Warning:
       This API is under development. Currently it is a no-operation (NOP)
       function on AMD GPUs and returns :py:obj:`~.hipSuccess`.

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - stream in which to enqueue the attach operation

       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - pointer to memory (must be a pointer to managed memory or
           to a valid host-accessible region of system-allocated memory)

       length (:py:obj:`~.int`) -- *IN*:
           - length of memory (defaults to zero)

       flags (:py:obj:`~.int`) -- *IN*:
           - must be one of hipMemAttachGlobal, hipMemAttachHost or
           hipMemAttachSingle (defaults to hipMemAttachSingle)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamAttachMemAsync(hipStream_t stream, void * dev_ptr, size_t length, unsigned int flags)


.. py:function:: hipMallocAsync(size, stream)

   Allocates memory with stream ordered semantics

   Inserts a memory allocation operation into ``stream.``
   A pointer to the allocated memory is returned immediately in *dptr.
   The allocation must not be accessed until the allocation operation completes.
   The allocation comes from the memory pool associated with the stream's device.

   Note:
       The default memory pool of a device contains device memory from that device.

   Note:
       Basic stream ordering allows future work submitted into the same stream to use the
       allocation. Stream query, stream synchronize, and HIP events can be used to guarantee that
       the allocation operation completes before work submitted in a separate stream runs.

   Note:
       During stream capture, this function results in the creation of an allocation node.
       In this case, the allocation is owned by the graph instead of the memory pool. The memory
       pool's properties are used to set the node's creation parameters.

   See:
       :py:obj:`~.hipMallocFromPoolAsync`, :py:obj:`~.hipFreeAsync`, :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolSetAccess`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           Number of bytes to allocate

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           The stream establishing the stream ordering contract and
           the memory pool to allocate from

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.rocm.bindings.util.types.DeviceArray`/:py:obj:`~.object`:
               Returned device pointer of memory allocation

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMallocAsync(void ** dev_ptr, size_t size, hipStream_t stream)


.. py:function:: hipFreeAsync(dev_ptr, stream)

   Frees memory with stream ordered semantics

   Inserts a free operation into ``stream.``
   The allocation must not be used after stream execution reaches the free.
   After this API returns, accessing the memory from any subsequent work launched on the GPU
   or querying its pointer attributes results in undefined behavior.

   Note:
       During stream capture, this function results in the creation of a free node and
       must therefore be passed the address of a graph allocation.

   See:
       :py:obj:`~.hipMallocFromPoolAsync`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolSetAccess`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to device memory to free

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           The stream, where the destruciton will occur according to the execution order

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFreeAsync(void * dev_ptr, hipStream_t stream)


.. py:function:: hipMemPoolTrimTo(mem_pool, min_bytes_to_hold)

   Releases freed memory back to the OS

   Releases memory back to the OS until the pool contains fewer than ``min_bytes_to_keep``
   reserved bytes, or there is no more memory that the allocator can safely release.
   The allocator cannot release OS allocations that back outstanding asynchronous allocations.
   The OS allocations may happen at different granularity from the user allocations.

   Note:
       Allocations that have not been freed count as outstanding.

   Note:
       Allocations that have been asynchronously freed but whose completion has
       not been observed on the host (eg. by a synchronize) can count as outstanding.

   See:
       :py:obj:`~.hipMallocFromPoolAsync`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipFreeAsync`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolSetAccess`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           The memory pool to trim allocations

       min_bytes_to_hold (:py:obj:`~.int`) -- *IN*:
           If the pool has less than min_bytes_to_hold reserved,
           then the TrimTo operation is a no-op.  Otherwise the memory pool will contain
           at least min_bytes_to_hold bytes reserved after the operation.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold)


.. py:function:: hipMemPoolSetAttribute(mem_pool, attr, value)

   Sets attributes of a memory pool

   Supported attributes are:
   - ``hipMemPoolAttrReleaseThreshold:`` (value type = cuuint64_t)
                                    Amount of reserved memory in bytes to hold onto before trying
                                    to release memory back to the OS. When more than the release
                                    threshold bytes of memory are held by the memory pool, the
                                    allocator will try to release memory back to the OS on the
                                    next call to stream, event or context synchronize. (default 0)
   - ``hipMemPoolReuseFollowEventDependencies:`` (value type = int)
                                    Allow ``hipMallocAsync`` to use memory asynchronously freed
                                    in another stream as long as a stream ordering dependency
                                    of the allocating stream on the free action exists.
                                    HIP events and null stream interactions can create the required
                                    stream ordered dependencies. (default enabled)
   - ``hipMemPoolReuseAllowOpportunistic:`` (value type = int)
                                    Allow reuse of already completed frees when there is no
   dependency between the free and allocation. (default enabled)
   - ``hipMemPoolReuseAllowInternalDependencies:`` (value type = int)
                                    Allow ``hipMallocAsync`` to insert new stream dependencies
                                    in order to establish the stream ordering required to reuse
                                    a piece of memory released by ``hipFreeAsync`` (default enabled).

   See:
       :py:obj:`~.hipMallocFromPoolAsync`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipFreeAsync`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAccess`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           The memory pool to modify

       attr (:py:obj:`~.hipMemPoolAttr`) -- *IN*:
           The attribute to modify

       value (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the value to assign

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void * value)


.. py:function:: hipMemPoolGetAttribute(mem_pool, attr, value)

   Gets attributes of a memory pool

   Supported attributes are:
   - ``hipMemPoolAttrReleaseThreshold:`` (value type = cuuint64_t)
                                    Amount of reserved memory in bytes to hold onto before trying
                                    to release memory back to the OS. When more than the release
                                    threshold bytes of memory are held by the memory pool, the
                                    allocator will try to release memory back to the OS on the
                                    next call to stream, event or context synchronize. (default 0)
   - ``hipMemPoolReuseFollowEventDependencies:`` (value type = int)
                                    Allow ``hipMallocAsync`` to use memory asynchronously freed
                                    in another stream as long as a stream ordering dependency
                                    of the allocating stream on the free action exists.
                                    HIP events and null stream interactions can create the required
                                    stream ordered dependencies. (default enabled)
   - ``hipMemPoolReuseAllowOpportunistic:`` (value type = int)
                                    Allow reuse of already completed frees when there is no
   dependency between the free and allocation. (default enabled)
   - ``hipMemPoolReuseAllowInternalDependencies:`` (value type = int)
                                    Allow ``hipMallocAsync`` to insert new stream dependencies
                                    in order to establish the stream ordering required to reuse
                                    a piece of memory released by ``hipFreeAsync`` (default enabled).

   See:
       :py:obj:`~.hipMallocFromPoolAsync`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipFreeAsync`,
       :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolSetAccess`,
       :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           The memory pool to get attributes of

       attr (:py:obj:`~.hipMemPoolAttr`) -- *IN*:
           The attribute to get

       value (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Retrieved value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void * value)


.. py:function:: hipMemPoolSetAccess(mem_pool, desc_list, count)

   Controls visibility of the specified pool between devices

   See:
       :py:obj:`~.hipMallocFromPoolAsync`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipFreeAsync`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           Memory pool for acccess change

       desc_list (:py:obj:`~.hipMemAccessDesc`/:py:obj:`~.object`) -- *IN*:
           Array of access descriptors. Each descriptor instructs the access to
           enable for a single gpu

       count (:py:obj:`~.int`) -- *IN*:
           Number of descriptors in the map array.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc * desc_list, size_t count)


.. py:function:: hipMemPoolGetAccess(mem_pool, location)

   Returns the accessibility of a pool from a device

   Returns the accessibility of the pool's memory from the specified location.

   See:
       :py:obj:`~.hipMallocFromPoolAsync`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipFreeAsync`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolSetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           Memory pool being queried

       location (:py:obj:`~.hipMemLocation`/:py:obj:`~.object`) -- *IN*:
           Location/device for memory pool access

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipMemAccessFlags`:
               Accessibility of the memory pool from the specified location/device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolGetAccess(hipMemAccessFlags * flags, hipMemPool_t mem_pool, hipMemLocation * location)


.. py:function:: hipMemPoolCreate(pool_props)

   Creates a memory pool

   Creates a HIP memory pool and returns the handle in ``mem_pool.`` The ``pool_props`` determines
   the properties of the pool such as the backing device and IPC capabilities.

   By default, the memory pool will be accessible from the device it is allocated on.

   Note:
       Specifying hipMemHandleTypeNone creates a memory pool that will not support IPC.

   See:
       :py:obj:`~.hipMallocFromPoolAsync`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipFreeAsync`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipMemPoolDestroy`, :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`,
       :py:obj:`~.hipMemPoolSetAccess`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       pool_props (:py:obj:`~.hipMemPoolProps`/:py:obj:`~.object`) -- *IN*:
           Memory pool properties

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.ihipMemPoolHandle_t`:
               Contains createed memory pool

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolCreate(hipMemPool_t * mem_pool, const hipMemPoolProps * pool_props)


.. py:function:: hipMemPoolDestroy(mem_pool)

   Destroys the specified memory pool

   If any pointers obtained from this pool haven't been freed or
   the pool has free operations that haven't completed
   when ``hipMemPoolDestroy`` is invoked, the function will return immediately and the
   resources associated with the pool will be released automatically
   once there are no more outstanding allocations.

   Destroying the current mempool of a device sets the default mempool of
   that device as the current mempool for that device.

   Note:
       A device's default memory pool cannot be destroyed.

   See:
       :py:obj:`~.hipMallocFromPoolAsync`, :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipFreeAsync`, :py:obj:`~.hipMemPoolGetAttribute`,
       :py:obj:`~.hipMemPoolCreate` :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`,
       :py:obj:`~.hipMemPoolSetAccess`, :py:obj:`~.hipMemPoolGetAccess`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           Memory pool for destruction

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool)


.. py:function:: hipMallocFromPoolAsync(size, mem_pool, stream)

   Allocates memory from a specified pool with stream ordered semantics.

   Inserts an allocation operation into ``stream.``
   A pointer to the allocated memory is returned immediately in ``dev_ptr.``
   The allocation must not be accessed until the allocation operation completes.
   The allocation comes from the specified memory pool.

   Note:
       The specified memory pool may be from a device different than that of the specified `stream`.

   Basic stream ordering allows future work submitted into the same stream to use the allocation.
   Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
   operation completes before work submitted in a separate stream runs.

   Note:
       During stream capture, this function results in the creation of an allocation node. In this
       case, the allocation is owned by the graph instead of the memory pool. The memory pool's
       properties are used to set the node's creation parameters.

   See:
       :py:obj:`~.hipMallocAsync`, :py:obj:`~.hipFreeAsync`, :py:obj:`~.hipMemPoolGetAttribute`, :py:obj:`~.hipMemPoolCreate`
       :py:obj:`~.hipMemPoolTrimTo`, :py:obj:`~.hipDeviceSetMemPool`, :py:obj:`~.hipMemPoolSetAttribute`, :py:obj:`~.hipMemPoolSetAccess`,
       :py:obj:`~.hipMemPoolGetAccess`,

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           Number of bytes to allocate

       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           The pool to allocate from

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           The stream establishing the stream ordering semantic

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.rocm.bindings.util.types.DeviceArray`/:py:obj:`~.object`:
               Returned device pointer

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMallocFromPoolAsync(void ** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream)


.. py:function:: hipMemPoolExportToShareableHandle(shared_handle, mem_pool, handle_type, flags)

   Exports a memory pool to the requested handle type.

   Given an IPC capable mempool, create an OS handle to share the pool with another process.
   A recipient process can convert the shareable handle into a mempool with `hipMemPoolImportFromShareableHandle`. Individual pointers can then be shared with the `hipMemPoolExportPointer` and ``hipMemPoolImportPointer`` APIs. The implementation of what the
   shareable handle is and how it can be transferred is defined by the requested handle type.

   Note:
       To create an IPC capable mempool, create a mempool with a ``hipMemAllocationHandleType``
       other than ``hipMemHandleTypeNone.``

   See:
       :py:obj:`~.hipMemPoolImportFromShareableHandle`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       shared_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to the location in which to store the requested handle

       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           Pool to export

       handle_type (:py:obj:`~.hipMemAllocationHandleType`) -- *IN*:
           The type of handle to create

       flags (:py:obj:`~.int`) -- *IN*:
           Must be 0

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorOutOfMemory`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolExportToShareableHandle(void * shared_handle, hipMemPool_t mem_pool, hipMemAllocationHandleType handle_type, unsigned int flags)


.. py:function:: hipMemPoolImportFromShareableHandle(shared_handle, handle_type, flags)

   Imports a memory pool from a shared handle.

   Specific allocations can be imported from the imported pool with ``hipMemPoolImportPointer.``

   Note:
       Imported memory pools do not support creating new allocations.
       As such imported memory pools may not be used in ``hipDeviceSetMemPool``
       or ``hipMallocFromPoolAsync`` calls.

   See:
       :py:obj:`~.hipMemPoolExportToShareableHandle`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       shared_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           OS handle of the pool to open

       handle_type (:py:obj:`~.hipMemAllocationHandleType`) -- *IN*:
           The type of handle being imported

       flags (:py:obj:`~.int`) -- *IN*:
           Must be 0

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.ihipMemPoolHandle_t`:
               Returned memory pool

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t * mem_pool, void * shared_handle, hipMemAllocationHandleType handle_type, unsigned int flags)


.. py:function:: hipMemPoolExportPointer(dev_ptr)

   Export data to share a memory pool allocation between processes.

   Constructs ``export_data`` for sharing a specific allocation from an already shared memory pool.
   The recipient process can import the allocation with the ``hipMemPoolImportPointer`` api.
   The data is not a handle and may be shared through any IPC mechanism.

   See:
       :py:obj:`~.hipMemPoolImportPointer`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to memory being exported

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.hipMemPoolPtrExportData`:
               Returned export data

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData * export_data, void * dev_ptr)


.. py:function:: hipMemPoolImportPointer(mem_pool, export_data)

   Import a memory pool allocation from another process.

   Returns in ``dev_ptr`` a pointer to the imported memory.
   The imported memory must not be accessed before the allocation operation completes
   in the exporting process. The imported memory must be freed from all importing processes before
   being freed in the exporting process. The pointer may be freed with ``hipFree``
   or ``hipFreeAsync.`` If ``hipFreeAsync`` is used, the free must be completed
   on the importing process before the free operation on the exporting process.

   Note:
       The ``hipFreeAsync`` api may be used in the exporting process before
       the ``hipFreeAsync`` operation completes in its stream as long as the
       ``hipFreeAsync`` in the exporting process specifies a stream with
       a stream dependency on the importing process's ``hipFreeAsync.``

   See:
       :py:obj:`~.hipMemPoolExportPointer`

   Warning:
       This API is marked as Beta. While this feature is complete, it can
       change and might have outstanding issues.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       mem_pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`) -- *IN*:
           Memory pool from which to import a pointer

       export_data (:py:obj:`~.hipMemPoolPtrExportData`/:py:obj:`~.object`) -- *IN*:
           Data specifying the memory to import

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to imported memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPoolImportPointer(void ** dev_ptr, hipMemPool_t mem_pool, hipMemPoolPtrExportData * export_data)


.. py:function:: hipMemSetMemPool(location, type, pool)

   Sets memory pool for memory location and allocation type.

   Args:
       location (:py:obj:`~.hipMemLocation`/:py:obj:`~.object`):
           (undocumented)

       type (:py:obj:`~.hipMemAllocationType`):
           (undocumented)

       pool (:py:obj:`~.ihipMemPoolHandle_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemSetMemPool(hipMemLocation * location, hipMemAllocationType type, hipMemPool_t pool)


.. py:function:: hipMemGetMemPool(location, type)

   Retrieves memory pool for memory location and allocation type.

   Args:
       location (:py:obj:`~.hipMemLocation`/:py:obj:`~.object`):
           (undocumented)

       type (:py:obj:`~.hipMemAllocationType`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * pool (:py:obj:`~.ihipMemPoolHandle_t`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemGetMemPool(hipMemPool_t * pool, hipMemLocation * location, hipMemAllocationType type)


.. py:function:: hipMemGetDefaultMemPool(location, type)

   Returns the default memory pool for a given location and allocation type

   Args:
       location (:py:obj:`~.hipMemLocation`/:py:obj:`~.object`) -- *IN*:
           location type for which to get the default memory pool,
           currently only hipMemLocationTypeDevice is supported

       type (:py:obj:`~.hipMemAllocationType`) -- *IN*:
           allocation type for which to get the default memory pool,
           currently only hipMemAllocationTypePinned & hipMemAllocationTypeManaged are supported

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipMemPoolHandle_t`:
               Returned memory pool

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemGetDefaultMemPool(hipMemPool_t * memPool, hipMemLocation * location, hipMemAllocationType type)


.. py:function:: hipHostAlloc(size, flags)

   Allocate device accessible page locked host memory

   If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.

   Flags:
   - :py:obj:`~.hipHostAllocDefault`   Default pinned memory allocation on the host.
   - :py:obj:`~.hipHostAllocPortable`  Memory is considered allocated by all contexts.
   - :py:obj:`~.hipHostAllocMapped`    Map the allocation into the address space for the current device.
   - :py:obj:`~.hipHostAllocWriteCombined`  Allocates the memory as write-combined.
   - :py:obj:`~.hipHostAllocUncached`  Allocate the host memory on extended fine grained access system
                            memory pool

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           Requested memory size in bytes

       flags (:py:obj:`~.int`) -- *IN*:
           Type of host memory allocation see below

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to the allocated host pinned memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipHostAlloc(void ** ptr, size_t size, unsigned int flags)


.. py:function:: hipHostGetDevicePointer(hstPtr, flags)

   Get Device pointer from Host Pointer allocated through hipHostMalloc

   See:
       :py:obj:`~.hipSetDeviceFlags`, :py:obj:`~.hipHostMalloc`

   Args:
       hstPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Host Pointer allocated through hipHostMalloc

       flags (:py:obj:`~.int`) -- *IN*:
           Flags to be passed for extension

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Device Pointer mapped to passed host pointer

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipHostGetDevicePointer(void ** devPtr, void * hstPtr, unsigned int flags)


.. py:function:: hipHostGetFlags(hostPtr)

   Return flags associated with host pointer

   See:
       :py:obj:`~.hipHostMalloc`

   Args:
       hostPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Host Pointer allocated through hipHostMalloc

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Memory location to store flags

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipHostGetFlags(unsigned int * flagsPtr, void * hostPtr)


.. py:function:: hipHostRegister(hostPtr, sizeBytes, flags)

   Register host memory so it can be accessed from the current device.

   Flags:
    - :py:obj:`~.hipHostRegisterDefault`   Memory is Mapped and Portable
    - :py:obj:`~.hipHostRegisterPortable`  Memory is considered registered by all contexts.  HIP only supports
   one context so this is always assumed true.
    - :py:obj:`~.hipHostRegisterMapped`    Map the allocation into the address space for the current device.
   The device pointer can be obtained with :py:obj:`~.hipHostGetDevicePointer`.
    - :py:obj:`~.hipExtHostRegisterUncached`  Map the host memory onto extended fine grained access system
   memory pool.

    After registering the memory, use :py:obj:`~.hipHostGetDevicePointer` to obtain the mapped device pointer.
    On many systems, the mapped device pointer will have a different value than the mapped host
   pointer.  Applications must use the device pointer in device code, and the host pointer in host
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

   See:
       :py:obj:`~.hipHostUnregister`, :py:obj:`~.hipHostGetFlags`, :py:obj:`~.hipHostGetDevicePointer`

   Args:
       hostPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to host memory to be registered.

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Size of the host memory

       flags (:py:obj:`~.int`) -- *IN*:
           See below.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipHostRegister(void * hostPtr, size_t sizeBytes, unsigned int flags)


.. py:function:: hipHostUnregister(hostPtr)

   Un-register host pointer

   See:
       :py:obj:`~.hipHostRegister`

   Args:
       hostPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Host pointer previously registered with :py:obj:`~.hipHostRegister`

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: Error code

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipHostUnregister(void * hostPtr)


.. py:function:: hipMallocPitch(width, height)

   Allocates at least width (in bytes) * height bytes of linear memory
    Padding may occur to ensure alighnment requirements are met for the given row
    The change in width size due to padding will be retu [...]

   Allocates at least width (in bytes) * height bytes of linear memory
   Padding may occur to ensure alighnment requirements are met for the given row
   The change in width size due to padding will be returned in *pitch.
   Currently the alignment is set to 128 bytes

   If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.

   See:
       :py:obj:`~.hipMalloc`, :py:obj:`~.hipFree`, :py:obj:`~.hipMallocArray`, :py:obj:`~.hipFreeArray`, :py:obj:`~.hipHostFree`, :py:obj:`~.hipMalloc3D`,
       :py:obj:`~.hipMalloc3DArray`, :py:obj:`~.hipHostMalloc`

   Args:
       width (:py:obj:`~.int`) -- *IN*:
           Requested pitched allocation width (in bytes)

       height (:py:obj:`~.int`) -- *IN*:
           Requested pitched allocation height

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: Error code
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to the allocated device memory
       * :py:obj:`~.int`:
               Pitch for allocation (in bytes)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMallocPitch(void ** ptr, size_t * pitch, size_t width, size_t height)


.. py:function:: hipMemAllocPitch(widthInBytes, height, elementSizeBytes)

   Allocates at least width (in bytes) * height bytes of linear memory
    Padding may occur to ensure alighnment requirements are met for the given row
    The change in width size due to padding will be retu [...]

   Allocates at least width (in bytes) * height bytes of linear memory
   Padding may occur to ensure alighnment requirements are met for the given row
   The change in width size due to padding will be returned in *pitch.
   Currently the alignment is set to 128 bytes

   If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
    The intended usage of pitch is as a separate parameter of the allocation, used to compute
   addresses within the 2D array. Given the row and column of an array element of type T, the
   address is computed as: T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;

   See:
       :py:obj:`~.hipMalloc`, :py:obj:`~.hipFree`, :py:obj:`~.hipMallocArray`, :py:obj:`~.hipFreeArray`, :py:obj:`~.hipHostFree`, :py:obj:`~.hipMalloc3D`,
       :py:obj:`~.hipMalloc3DArray`, :py:obj:`~.hipHostMalloc`

   Args:
       widthInBytes (:py:obj:`~.int`) -- *IN*:
           Requested pitched allocation width (in bytes)

       height (:py:obj:`~.int`) -- *IN*:
           Requested pitched allocation height

       elementSizeBytes (:py:obj:`~.int`) -- *IN*:
           The size of element bytes, should be 4, 8 or 16

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: Error code
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to the allocated device memory
       * :py:obj:`~.int`:
               Pitch for allocation (in bytes)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemAllocPitch(hipDeviceptr_t * dptr, size_t * pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes)


.. py:function:: hipFree(ptr)

   Free memory allocated by the HIP-Clang hip memory allocation API.
    This API performs an implicit hipDeviceSynchronize() call.
    If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.

   See:
       :py:obj:`~.hipMalloc`, :py:obj:`~.hipMallocPitch`, :py:obj:`~.hipMallocArray`, :py:obj:`~.hipFreeArray`, :py:obj:`~.hipHostFree`, :py:obj:`~.hipMalloc3D`,
       :py:obj:`~.hipMalloc3DArray`, :py:obj:`~.hipHostMalloc`

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to memory to be freed

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: One of:
               - py:obj:`~.hipSuccess`
               - py:obj:`~.hipErrorInvalidDevicePointer` (if pointer is invalid, including host pointers allocated
                     with hipHostMalloc)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFree(void * ptr)


.. py:function:: hipFreeHost(ptr)

   Frees page-locked memory
   This API performs an implicit hipDeviceSynchronize() call.
   If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to memory to be freed

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`,
                   :py:obj:`~.hipErrorInvalidValue` (if pointer is invalid, including device pointers allocated
           with hipMalloc)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFreeHost(void * ptr)


.. py:function:: hipHostFree(ptr)

   Free memory allocated by the HIP-Clang hip host memory allocation API
    This API performs an implicit hipDeviceSynchronize() call.
    If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.

   See:
       :py:obj:`~.hipMalloc`, :py:obj:`~.hipMallocPitch`, :py:obj:`~.hipFree`, :py:obj:`~.hipMallocArray`, :py:obj:`~.hipFreeArray`, :py:obj:`~.hipMalloc3D`,
       :py:obj:`~.hipMalloc3DArray`, :py:obj:`~.hipHostMalloc`

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to memory to be freed

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`,
                    :py:obj:`~.hipErrorInvalidValue` (if pointer is invalid, including device pointers allocated with
           hipMalloc)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipHostFree(void * ptr)


.. py:function:: hipMemcpy(dst, src, sizeBytes, kind)

   Copy data from src to dst.

   It supports memory from host to device,
    device to host, device to device and host to host
    The src and dst must not overlap.

    For hipMemcpy, the copy is always performed by the current device (set by hipSetDevice).
    For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the
    device where the src data is physically located. For optimal peer-to-peer copies, the copy
   device must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with
   copy agent as the current device and src/dst as the peerDevice argument.  if this is not done,
   the hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
    Calling hipMemcpy with dst and src pointers that do not match the hipMemcpyKind results in
    undefined behavior.

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being copy to

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data being copy from

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Kind of transfer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorUnknown`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind)


.. py:function:: hipMemcpyWithStream(dst, src, sizeBytes, kind, stream)

   Memory copy on the stream.
    It allows single or multiple devices to do memory copy on single or multiple streams.
    The operation is akin to hipMemcpyAsync + hipStreamSynchronize.
    Since it is a sync API, it is not allowed during graph capture.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipStreamCreate`, :py:obj:`~.hipStreamSynchronize`, :py:obj:`~.hipStreamDestroy`, :py:obj:`~.hipSetDevice`,
       :py:obj:`~.hipLaunchKernelGGL`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being copy to

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data being copy from

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Kind of transfer

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Valid stream

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorUnknown`, :py:obj:`~.hipErrorContextIsDestroyed`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyWithStream(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpyHtoD(dst, src, sizeBytes)

   Copy data from Host to Device

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being copy to

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data being copy from

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, const void * src, size_t sizeBytes)


.. py:function:: hipMemcpyDtoH(dst, src, sizeBytes)

   Copy data from Device to Host

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being copy to

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data being copy from

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyDtoH(void * dst, hipDeviceptr_t src, size_t sizeBytes)


.. py:function:: hipMemcpyDtoD(dst, src, sizeBytes)

   Copy data from Device to Device

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being copy to

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data being copy from

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes)


.. py:function:: hipMemcpyAtoD(dstDevice, srcArray, srcOffset, ByteCount)

   Copies from one 1D array to device memory.

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dstDevice (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Destination device pointer

       srcArray (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Source array

       srcOffset (:py:obj:`~.int`) -- *IN*:
           Offset in bytes of source array

       ByteCount (:py:obj:`~.int`) -- *IN*:
           Size of memory copy in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, size_t srcOffset, size_t ByteCount)


.. py:function:: hipMemcpyDtoA(dstArray, dstOffset, srcDevice, ByteCount)

   Copies from device memory to a 1D array.

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dstArray (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *OUT*:
           Destination array

       dstOffset (:py:obj:`~.int`) -- *IN*:
           Offset in bytes of destination array

       srcDevice (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Source device pointer

       ByteCount (:py:obj:`~.int`) -- *IN*:
           Size of memory copy in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset, hipDeviceptr_t srcDevice, size_t ByteCount)


.. py:function:: hipMemcpyAtoA(dstArray, dstOffset, srcArray, srcOffset, ByteCount)

   Copies from one 1D array to another.

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dstArray (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *OUT*:
           Destination array

       dstOffset (:py:obj:`~.int`) -- *IN*:
           Offset in bytes of destination array

       srcArray (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Source array

       srcOffset (:py:obj:`~.int`) -- *IN*:
           Offset in bytes of source array

       ByteCount (:py:obj:`~.int`) -- *IN*:
           Size of memory copy in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray, size_t srcOffset, size_t ByteCount)


.. py:function:: hipMemcpyHtoDAsync(dst, src, sizeBytes, stream)

   Copy data from Host to Device asynchronously

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being copy to

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data being copy from

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void * src, size_t sizeBytes, hipStream_t stream)


.. py:function:: hipMemcpyDtoHAsync(dst, src, sizeBytes, stream)

   Copy data from Device to Host asynchronously

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being copy to

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data being copy from

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyDtoHAsync(void * dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)


.. py:function:: hipMemcpyDtoDAsync(dst, src, sizeBytes, stream)

   Copy data from Device to Device asynchronously

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being copy to

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data being copy from

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)


.. py:function:: hipMemcpyAtoHAsync(dstHost, srcArray, srcOffset, ByteCount, stream)

   Copies from one 1D array to host memory.

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dstHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Destination pointer

       srcArray (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Source array

       srcOffset (:py:obj:`~.int`) -- *IN*:
           Offset in bytes of source array

       ByteCount (:py:obj:`~.int`) -- *IN*:
           Size of memory copy in bytes

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyAtoHAsync(void * dstHost, hipArray_t srcArray, size_t srcOffset, size_t ByteCount, hipStream_t stream)


.. py:function:: hipMemcpyHtoAAsync(dstArray, dstOffset, srcHost, ByteCount, stream)

   Copies from host memory to a 1D array.

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`, :py:obj:`~.hipMemAllocHost`,
       :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`, :py:obj:`~.hipMemcpyAtoA`,
       :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`, :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`,
       :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`, :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`,
       :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`, :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`,
       :py:obj:`~.hipMemHostAlloc`, :py:obj:`~.hipMemHostGetDevicePointer`

   Args:
       dstArray (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *OUT*:
           Destination array

       dstOffset (:py:obj:`~.int`) -- *IN*:
           Offset in bytes of destination array

       srcHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Source host pointer

       ByteCount (:py:obj:`~.int`) -- *IN*:
           Size of memory copy in bytes

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset, const void * srcHost, size_t ByteCount, hipStream_t stream)


.. py:function:: hipModuleGetGlobal(hmod, name)

   Returns a global pointer from a module.

   Returns in *dptr and *bytes the pointer and size of the global of name name located in module
   hmod. If no variable of that name exists, it returns hipErrorNotFound. Both parameters dptr and
   bytes are optional. If one of them is NULL, it is ignored and hipSuccess is returned.

   Args:
       hmod (:py:obj:`~.ihipModule_t`/:py:obj:`~.object`) -- *IN*:
           Module to retrieve global from

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Name of global to retrieve

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotFound`, :py:obj:`~.hipErrorInvalidContext`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Returns global device pointer
       * :py:obj:`~.int`:
               Returns global size in bytes

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleGetGlobal(hipDeviceptr_t * dptr, size_t * bytes, hipModule_t hmod, const char * name)


.. py:function:: hipGetSymbolAddress(symbol)

   Gets device pointer associated with symbol on the device.

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the symbole of the device

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               pointer to the device associated the symbole

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetSymbolAddress(void ** devPtr, const void * symbol)


.. py:function:: hipGetSymbolSize(symbol)

   Gets the size of the given symbol on the device.

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the device symbole

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               pointer to the size

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetSymbolSize(size_t * size, const void * symbol)


.. py:function:: hipGetProcAddress(symbol, hipVersion, flags)

   Gets the pointer of requested HIP driver function.

   Returns hipSuccess if the returned pfn is addressed to the pointer of found driver function.

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           The Symbol name of the driver function to request.

       hipVersion (:py:obj:`~.int`) -- *IN*:
           The HIP version for the requested driver function symbol.
           HIP version is defined as 100*version_major + version_minor. For example, in HIP 6.1, the
           hipversion is 601, for the symbol function "hipGetDeviceProperties", the specified hipVersion 601
           is greater or equal to the version 600, the symbol function will be handle properly as backend
           compatible function.

       flags (:py:obj:`~.int`) -- *IN*:
           Currently only default flag is suppported.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`.
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Output pointer to the requested driver function.
       * :py:obj:`~.hipDriverProcAddressQueryResult`:
               Optional enumeration for returned status of searching for symbol driver
               function based on the input hipVersion.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetProcAddress(const char * symbol, void ** pfn, int hipVersion, uint64_t flags, hipDriverProcAddressQueryResult * symbolStatus)


.. py:function:: hipMemcpyToSymbol(symbol, src, sizeBytes, offset, kind)

   Copies data to the given symbol on the device.
   Symbol HIP APIs allow a kernel to define a device-side data symbol which can be accessed on
   the host side. The symbol can be in __constant or device space.
   Note that the symbol name needs to be encased in the HIP_SYMBOL macro.
   This also applies to hipMemcpyFromSymbol, hipGetSymbolAddress, and hipGetSymbolSize.
   For detailed usage, see the
   <a
   href="https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html:py:obj:`~.memcpytosymbol`">memcpyToSymbol
   example</a> in the HIP Porting Guide.

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           pointer to the device symbole

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the source address

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           size in bytes to copy

       offset (:py:obj:`~.int`) -- *IN*:
           offset in bytes from start of symbole

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           type of memory transfer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyToSymbol(const void * symbol, const void * src, size_t sizeBytes, size_t offset, hipMemcpyKind kind)


.. py:function:: hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind, stream)

   Copies data to the given symbol on the device asynchronously.

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           pointer to the device symbole

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the source address

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           size in bytes to copy

       offset (:py:obj:`~.int`) -- *IN*:
           offset in bytes from start of symbole

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           type of memory transfer

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyToSymbolAsync(const void * symbol, const void * src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpyFromSymbol(dst, symbol, sizeBytes, offset, kind)

   Copies data from the given symbol on the device.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Returns pointer to destinition memory address

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the symbole address on the device

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Size in bytes to copy

       offset (:py:obj:`~.int`) -- *IN*:
           Offset in bytes from the start of symbole

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of memory transfer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyFromSymbol(void * dst, const void * symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind)


.. py:function:: hipMemcpyFromSymbolAsync(dst, symbol, sizeBytes, offset, kind, stream)

   Copies data from the given symbol on the device asynchronously.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Returns pointer to destinition memory address

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the symbole address on the device

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           size in bytes to copy

       offset (:py:obj:`~.int`) -- *IN*:
           offset in bytes from the start of symbole

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           type of memory transfer

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpyAsync(dst, src, sizeBytes, kind, stream)

   Copies data from src to dst asynchronously.

   The copy is always performed by the device associated with the specified stream.

    For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is
   attached to the device where the src data is physically located.
    For optimal peer-to-peer copies, the copy device must be able to access the src and dst
   pointers (by calling hipDeviceEnablePeerAccess) with copy agent as the current device and
   src/dest as the peerDevice argument. If enabling device peer access is not done, the memory copy
   will still work, but will perform the copy using a staging buffer on the host.

   Note:
       If host or dst are not pinned, the memory copy will be performed synchronously. For
       best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyToArray`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpyFromArray`,
       :py:obj:`~.hipMemcpy2DFromArray`, :py:obj:`~.hipMemcpyArrayToArray`, :py:obj:`~.hipMemcpy2DArrayToArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyFromSymbol`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpyToArrayAsync`, :py:obj:`~.hipMemcpy2DToArrayAsync`,
       :py:obj:`~.hipMemcpyFromArrayAsync`, :py:obj:`~.hipMemcpy2DFromArrayAsync`, :py:obj:`~.hipMemcpyToSymbolAsync`,
       :py:obj:`~.hipMemcpyFromSymbolAsync`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being copy to

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data being copy from

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of memory transfer

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorUnknown`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyAsync(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemset(dst, value, sizeBytes)

   Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
   byte value value.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being filled

       value (:py:obj:`~.int`) -- *IN*:
           Value to be set

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Data size in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset(void * dst, int value, size_t sizeBytes)


.. py:function:: hipMemsetD8(dest, value, count)

   Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
   byte value value.

   Args:
       dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data ptr to be filled

       value (:py:obj:`~.int`) -- *IN*:
           Value to be set

       count (:py:obj:`~.int`) -- *IN*:
           Number of values to be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count)


.. py:function:: hipMemsetD8Async(dest, value, count, stream)

   Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
   byte value value.

   hipMemsetD8Async() is asynchronous with respect to the host, so the call may return before the
   memset is complete. The operation can optionally be associated to a stream by passing a non-zero
   stream argument. If stream is non-zero, the operation may overlap with operations in other
   streams.

   Args:
       dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data ptr to be filled

       value (:py:obj:`~.int`) -- *IN*:
           Constant value to be set

       count (:py:obj:`~.int`) -- *IN*:
           Number of values to be set

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count, hipStream_t stream)


.. py:function:: hipMemsetD16(dest, value, count)

   Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
   short value value.

   Args:
       dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data ptr to be filled

       value (:py:obj:`~.int`) -- *IN*:
           Constant value to be set

       count (:py:obj:`~.int`) -- *IN*:
           Number of values to be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count)


.. py:function:: hipMemsetD16Async(dest, value, count, stream)

   Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
   short value value.

   hipMemsetD16Async() is asynchronous with respect to the host, so the call may return before the
   memset is complete. The operation can optionally be associated to a stream by passing a non-zero
   stream argument. If stream is non-zero, the operation may overlap with operations in other
   streams.

   Args:
       dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data ptr to be filled

       value (:py:obj:`~.int`) -- *IN*:
           Constant value to be set

       count (:py:obj:`~.int`) -- *IN*:
           Number of values to be set

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count, hipStream_t stream)


.. py:function:: hipMemsetD32(dest, value, count)

   Fills the memory area pointed to by dest with the constant integer
   value for specified number of times.

   Args:
       dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Data being filled

       value (:py:obj:`~.int`) -- *IN*:
           Constant value to be set

       count (:py:obj:`~.int`) -- *IN*:
           Number of values to be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count)


.. py:function:: hipMemsetAsync(dst, value, sizeBytes, stream)

   Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant
   byte value value.

   hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the
   memset is complete. The operation can optionally be associated to a stream by passing a non-zero
   stream argument. If stream is non-zero, the operation may overlap with operations in other
   streams.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to device memory

       value (:py:obj:`~.int`) -- *IN*:
           Value to set for each byte of specified memory

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           Size in bytes to set

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetAsync(void * dst, int value, size_t sizeBytes, hipStream_t stream)


.. py:function:: hipMemsetD32Async(dst, value, count, stream)

   Fills the memory area pointed to by dev with the constant integer
   value for specified number of times.

   hipMemsetD32Async() is asynchronous with respect to the host, so the call may return before the
   memset is complete. The operation can optionally be associated to a stream by passing a non-zero
   stream argument. If stream is non-zero, the operation may overlap with operations in other
   streams.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to device memory

       value (:py:obj:`~.int`) -- *IN*:
           Value to set for each byte of specified memory

       count (:py:obj:`~.int`) -- *IN*:
           Number of values to be set

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream)


.. py:function:: hipMemset2D(dst, pitch, value, width, height)

   Fills the memory area pointed to by dst with the constant value.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to 2D device memory

       pitch (:py:obj:`~.int`) -- *IN*:
           Pitch size in bytes of 2D device memory, unused if height equals 1

       value (:py:obj:`~.int`) -- *IN*:
           Constant value to set for each byte of specified memory

       width (:py:obj:`~.int`) -- *IN*:
           Width size in bytes in 2D memory

       height (:py:obj:`~.int`) -- *IN*:
           Height size in bytes in 2D memory

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset2D(void * dst, size_t pitch, int value, size_t width, size_t height)


.. py:function:: hipMemset2DAsync(dst, pitch, value, width, height, stream)

   Fills asynchronously the memory area pointed to by dst with the constant value.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to 2D device memory

       pitch (:py:obj:`~.int`) -- *IN*:
           Pitch size in bytes of 2D device memory, unused if height equals 1

       value (:py:obj:`~.int`) -- *IN*:
           Value to set for each byte of specified memory

       width (:py:obj:`~.int`) -- *IN*:
           Width size in bytes in 2D memory

       height (:py:obj:`~.int`) -- *IN*:
           Height size in bytes in 2D memory

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset2DAsync(void * dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream)


.. py:function:: hipMemset3D(pitchedDevPtr, value, extent)

   Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.

   Args:
       pitchedDevPtr (:py:obj:`~.hipPitchedPtr`) -- *IN*:
           Pointer to pitched device memory

       value (:py:obj:`~.int`) -- *IN*:
           Value to set for each byte of specified memory

       extent (:py:obj:`~.hipExtent`) -- *IN*:
           Size parameters for width field in bytes in device memory

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent)


.. py:function:: hipMemset3DAsync(pitchedDevPtr, value, extent, stream)

   Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.

   Args:
       pitchedDevPtr (:py:obj:`~.hipPitchedPtr`) -- *IN*:
           Pointer to pitched device memory

       value (:py:obj:`~.int`) -- *IN*:
           Value to set for each byte of specified memory

       extent (:py:obj:`~.hipExtent`) -- *IN*:
           Size parameters for width field in bytes in device memory

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent, hipStream_t stream)


.. py:function:: hipMemsetD2D8(dst, dstPitch, value, width, height)

   Fills 2D memory range of 'width' 8-bit values synchronously to the specified char value.
   Height specifies numbers of rows to set and dstPitch speicifies the number of bytes between each
   row.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to device memory

       dstPitch (:py:obj:`~.int`) -- *IN*:
           Pitch of dst device pointer

       value (:py:obj:`~.int`) -- *IN*:
           value to set

       width (:py:obj:`~.int`) -- *IN*:
           Width of row

       height (:py:obj:`~.int`) -- *IN*:
           Number of rows

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch, unsigned char value, size_t width, size_t height)


.. py:function:: hipMemsetD2D8Async(dst, dstPitch, value, width, height, stream)

   Fills 2D memory range of 'width' 8-bit values asynchronously to the specified char value.
   Height specifies numbers of rows to set and dstPitch speicifies the number of bytes between each
   row.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to device memory

       dstPitch (:py:obj:`~.int`) -- *IN*:
           Pitch of dst device pointer

       value (:py:obj:`~.int`) -- *IN*:
           value to set

       width (:py:obj:`~.int`) -- *IN*:
           Width of row

       height (:py:obj:`~.int`) -- *IN*:
           Number of rows

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream Identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD2D8Async(hipDeviceptr_t dst, size_t dstPitch, unsigned char value, size_t width, size_t height, hipStream_t stream)


.. py:function:: hipMemsetD2D16(dst, dstPitch, value, width, height)

   Fills 2D memory range of 'width' 16-bit values synchronously to the specified short
   value. Height specifies numbers of rows to set and dstPitch speicifies the number of bytes
   between each row.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to device memory

       dstPitch (:py:obj:`~.int`) -- *IN*:
           Pitch of dst device pointer

       value (:py:obj:`~.int`) -- *IN*:
           value to set

       width (:py:obj:`~.int`) -- *IN*:
           Width of row

       height (:py:obj:`~.int`) -- *IN*:
           Number of rows

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch, unsigned short value, size_t width, size_t height)


.. py:function:: hipMemsetD2D16Async(dst, dstPitch, value, width, height, stream)

   Fills 2D memory range of 'width' 16-bit values asynchronously to the specified short
   value. Height specifies numbers of rows to set and dstPitch speicifies the number of bytes
   between each row.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to device memory

       dstPitch (:py:obj:`~.int`) -- *IN*:
           Pitch of dst device pointer

       value (:py:obj:`~.int`) -- *IN*:
           value to set

       width (:py:obj:`~.int`) -- *IN*:
           Width of row

       height (:py:obj:`~.int`) -- *IN*:
           Number of rows

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream Identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch, unsigned short value, size_t width, size_t height, hipStream_t stream)


.. py:function:: hipMemsetD2D32(dst, dstPitch, value, width, height)

   Fills 2D memory range of 'width' 32-bit values synchronously to the specified int value.
   Height specifies numbers of rows to set and dstPitch speicifies the number of bytes between each
   row.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to device memory

       dstPitch (:py:obj:`~.int`) -- *IN*:
           Pitch of dst device pointer

       value (:py:obj:`~.int`) -- *IN*:
           value to set

       width (:py:obj:`~.int`) -- *IN*:
           Width of row

       height (:py:obj:`~.int`) -- *IN*:
           Number of rows

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch, unsigned int value, size_t width, size_t height)


.. py:function:: hipMemsetD2D32Async(dst, dstPitch, value, width, height, stream)

   Fills 2D memory range of 'width' 32-bit values asynchronously to the specified int
   value. Height specifies numbers of rows to set and dstPitch speicifies the number of bytes
   between each row.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to device memory

       dstPitch (:py:obj:`~.int`) -- *IN*:
           Pitch of dst device pointer

       value (:py:obj:`~.int`) -- *IN*:
           value to set

       width (:py:obj:`~.int`) -- *IN*:
           Width of row

       height (:py:obj:`~.int`) -- *IN*:
           Number of rows

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream Identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetD2D32Async(hipDeviceptr_t dst, size_t dstPitch, unsigned int value, size_t width, size_t height, hipStream_t stream)


.. py:function:: hipMemGetInfo()

   Query memory info.

   On ROCM, this function gets the actual free memory left on the current device, so supports
   the cases while running multi-workload (such as multiple processes, multiple threads, and
   multiple GPUs).

   Warning:
       On Windows, the free memory only accounts for memory allocated by this process and may
       be optimistic.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Returns free memory on the current device in bytes
       * :py:obj:`~.int`:
               Returns total allocatable memory on the current device in bytes

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemGetInfo(size_t * free, size_t * total)


.. py:function:: hipMemPtrGetInfo(ptr)

   Get allocated memory size via memory pointer.

   This function gets the allocated shared virtual memory size from memory pointer.

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to allocated memory

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Returns the allocated memory size in bytes

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemPtrGetInfo(void * ptr, size_t * size)


.. py:function:: hipMallocArray(desc, width, height, flags)

   Allocate an array on the device.

   See:
       :py:obj:`~.hipMalloc`, :py:obj:`~.hipMallocPitch`, :py:obj:`~.hipFree`, :py:obj:`~.hipFreeArray`, :py:obj:`~.hipHostMalloc`, :py:obj:`~.hipHostFree`

   Args:
       desc (:py:obj:`~.hipChannelFormatDesc`/:py:obj:`~.object`) -- *IN*:
           Requested channel format

       width (:py:obj:`~.int`) -- *IN*:
           Requested array allocation width

       height (:py:obj:`~.int`) -- *IN*:
           Requested array allocation height

       flags (:py:obj:`~.int`) -- *IN*:
           Requested properties of allocated array

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.hipArray`:
               Pointer to allocated array in device memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMallocArray(hipArray_t * array, const hipChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags)


.. py:function:: hipArrayCreate(pAllocateArray)

   Create an array memory pointer on the device.

   See:
       :py:obj:`~.hipMallocArray`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipFreeArray`

   Args:
       pAllocateArray (:py:obj:`~.HIP_ARRAY_DESCRIPTOR`/:py:obj:`~.object`) -- *IN*:
           Requested array desciptor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipArray`:
               Pointer to the array memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipArrayCreate(hipArray_t * pHandle, const HIP_ARRAY_DESCRIPTOR * pAllocateArray)


.. py:function:: hipArrayDestroy(array)

   Destroy an array memory pointer on the device.

   See:
       :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipFreeArray`

   Args:
       array (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Pointer to the array memory

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipArrayDestroy(hipArray_t array)


.. py:function:: hipArray3DCreate(pAllocateArray)

   Create a 3D array memory pointer on the device.

   See:
       :py:obj:`~.hipMallocArray`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipFreeArray`

   Args:
       pAllocateArray (:py:obj:`~.HIP_ARRAY3D_DESCRIPTOR`/:py:obj:`~.object`) -- *IN*:
           Requested array desciptor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipArray`:
               Pointer to the 3D array memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipArray3DCreate(hipArray_t * array, const HIP_ARRAY3D_DESCRIPTOR * pAllocateArray)


.. py:function:: hipMalloc3D(extent)

   Create a 3D memory pointer on the device.

   See:
       :py:obj:`~.hipMallocPitch`, :py:obj:`~.hipMemGetInfo`, :py:obj:`~.hipFree`

   Args:
       extent (:py:obj:`~.hipExtent`) -- *IN*:
           Requested extent

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipPitchedPtr`:
               Pointer to the 3D memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMalloc3D(hipPitchedPtr * pitchedDevPtr, hipExtent extent)


.. py:function:: hipFreeArray(array)

   Frees an array on the device.

   See:
       :py:obj:`~.hipMalloc`, :py:obj:`~.hipMallocPitch`, :py:obj:`~.hipFree`, :py:obj:`~.hipMallocArray`, :py:obj:`~.hipHostMalloc`, :py:obj:`~.hipHostFree`

   Args:
       array (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Pointer to array to free

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotInitialized`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFreeArray(hipArray_t array)


.. py:function:: hipMalloc3DArray(desc, extent, flags)

   Allocate an array on the device.

   See:
       :py:obj:`~.hipMalloc`, :py:obj:`~.hipMallocPitch`, :py:obj:`~.hipFree`, :py:obj:`~.hipFreeArray`, :py:obj:`~.hipHostMalloc`, :py:obj:`~.hipHostFree`

   Args:
       desc (:py:obj:`~.hipChannelFormatDesc`/:py:obj:`~.object`) -- *IN*:
           Requested channel format

       extent (:py:obj:`~.hipExtent`) -- *IN*:
           Requested array allocation width, height and depth

       flags (:py:obj:`~.int`) -- *IN*:
           Requested properties of allocated array

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.hipArray`:
               Pointer to allocated array in device memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMalloc3DArray(hipArray_t * array, const struct hipChannelFormatDesc * desc, struct hipExtent extent, unsigned int flags)


.. py:function:: hipArrayGetInfo(array)

   Gets info about the specified array

   See:
       :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipArray3DGetDescriptor`

   Args:
       array (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           - The HIP array to get info for

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue` :py:obj:`~.hipErrorInvalidHandle`
       * :py:obj:`~.hipChannelFormatDesc`:
               - Returned array type
       * :py:obj:`~.hipExtent`:
               - Returned array shape. 2D arrays will have depth of zero
       * :py:obj:`~.int`:
               - Returned array flags

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipArrayGetInfo(hipChannelFormatDesc * desc, hipExtent * extent, unsigned int * flags, hipArray_t array)


.. py:function:: hipArrayGetDescriptor(array)

   Gets a 1D or 2D array descriptor

   See:
       :py:obj:`~.hipArray3DCreate`, :py:obj:`~.hipArray3DGetDescriptor`, :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipMemAlloc`,
       :py:obj:`~.hipMemAllocHost`, :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`,
       :py:obj:`~.hipMemcpy3D`, :py:obj:`~.hipMemcpy3DAsync`, :py:obj:`~.hipMemcpyAtoA`, :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`,
       :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`, :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`,
       :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`, :py:obj:`~.hipMemcpyHtoD`, :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`,
       :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`, :py:obj:`~.hipMemHostAlloc`,
       :py:obj:`~.hipMemHostGetDevicePointer`, :py:obj:`~.hipMemsetD8`, :py:obj:`~.hipMemsetD16`, :py:obj:`~.hipMemsetD32`, :py:obj:`~.hipArrayGetInfo`

   Args:
       array (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           - Array to get descriptor of

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue` :py:obj:`~.hipErrorInvalidHandle`
       * :py:obj:`~.HIP_ARRAY_DESCRIPTOR`:
               - Returned array descriptor

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR * pArrayDescriptor, hipArray_t array)


.. py:function:: hipArray3DGetDescriptor(array)

   Gets a 3D array descriptor

   See:
       :py:obj:`~.hipArray3DCreate`, :py:obj:`~.hipArrayCreate`, :py:obj:`~.hipArrayDestroy`, :py:obj:`~.hipArrayGetDescriptor`, :py:obj:`~.hipMemAlloc`,
       :py:obj:`~.hipMemAllocHost`, :py:obj:`~.hipMemAllocPitch`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpy2DAsync`, :py:obj:`~.hipMemcpy2DUnaligned`,
       :py:obj:`~.hipMemcpy3D`, :py:obj:`~.hipMemcpy3DAsync`, :py:obj:`~.hipMemcpyAtoA`, :py:obj:`~.hipMemcpyAtoD`, :py:obj:`~.hipMemcpyAtoH`, :py:obj:`~.hipMemcpyAtoHAsync`,
       :py:obj:`~.hipMemcpyDtoA`, :py:obj:`~.hipMemcpyDtoD`, :py:obj:`~.hipMemcpyDtoDAsync`, :py:obj:`~.hipMemcpyDtoH`, :py:obj:`~.hipMemcpyDtoHAsync`,
       :py:obj:`~.hipMemcpyHtoA`, :py:obj:`~.hipMemcpyHtoAAsync`, :py:obj:`~.hipMemcpyHtoD`, :py:obj:`~.hipMemcpyHtoDAsync`, :py:obj:`~.hipMemFree`,
       :py:obj:`~.hipMemFreeHost`, :py:obj:`~.hipMemGetAddressRange`, :py:obj:`~.hipMemGetInfo`, :py:obj:`~.hipMemHostAlloc`,
       :py:obj:`~.hipMemHostGetDevicePointer`, :py:obj:`~.hipMemsetD8`, :py:obj:`~.hipMemsetD16`, :py:obj:`~.hipMemsetD32`, :py:obj:`~.hipArrayGetInfo`

   Args:
       array (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           - 3D array to get descriptor of

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidValue` :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorContextIsDestroyed`
       * :py:obj:`~.HIP_ARRAY3D_DESCRIPTOR`:
               - Returned 3D array descriptor

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR * pArrayDescriptor, hipArray_t array)


.. py:function:: hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind)

   Copies data between host and device.

   hipMemcpy2D supports memory matrix copy from the pointed area src to the pointed area dst.
   The copy direction is defined by kind which must be one of :py:obj:`~.hipMemcpyHostToDevice`,
   :py:obj:`~.hipMemcpyHostToDevice`, :py:obj:`~.hipMemcpyDeviceToHost` :py:obj:`~.hipMemcpyDeviceToDevice` or :py:obj:`~.hipMemcpyDefault`.
   Device to Device copies don't need to wait for host synchronization.
   The copy is executed on the default null tream. The src and dst must not overlap.
   dpitch and spitch are the widths in bytes in memory matrix, width cannot exceed dpitch or
   spitch.

   For hipMemcpy2D, the copy is always performed by the current device (set by hipSetDevice).
   For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the
   device where the src data is physically located. For optimal peer-to-peer copies, the copy device
   must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy
   agent as the current device and src/dst as the peerDevice argument.  if this is not done, the
   hipMemcpy2D will still work, but will perform the copy using a staging buffer on the host.

   Warning:
       Calling hipMemcpy2D with dst and src pointers that do not match the hipMemcpyKind
       results in undefined behavior.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpyToArray`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       dpitch (:py:obj:`~.int`) -- *IN*:
           Pitch size in bytes of destination memory

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Source memory address

       spitch (:py:obj:`~.int`) -- *IN*:
           Pitch size in bytes of source memory

       width (:py:obj:`~.int`) -- *IN*:
           Width size in bytes of matrix transfer (columns)

       height (:py:obj:`~.int`) -- *IN*:
           Height size in bytes of matrix transfer (rows)

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of transfer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind)


.. py:function:: hipMemcpyParam2D(pCopy)

   Copies memory for 2D arrays.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyToArray`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpyFromArray`,
       :py:obj:`~.hipMemcpyToSymbol`, :py:obj:`~.hipMemcpyAsync`

   Args:
       pCopy (:py:obj:`~.hip_Memcpy2D`/:py:obj:`~.object`) -- *IN*:
           Parameters for the memory copy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyParam2D(const hip_Memcpy2D * pCopy)


.. py:function:: hipMemcpyParam2DAsync(pCopy, stream)

   Copies memory for 2D arrays.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyToArray`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpyFromArray`,
       :py:obj:`~.hipMemcpyToSymbol`, :py:obj:`~.hipMemcpyAsync`

   Args:
       pCopy (:py:obj:`~.hip_Memcpy2D`/:py:obj:`~.object`) -- *IN*:
           Parameters for the memory copy

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to use

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D * pCopy, hipStream_t stream)


.. py:function:: hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)

   Copies data between host and device asynchronously.

   hipMemcpy2DAsync supports memory matrix copy from the pointed area src to the pointed area dst.
   The copy direction is defined by kind which must be one of :py:obj:`~.hipMemcpyHostToDevice`,
   :py:obj:`~.hipMemcpyDeviceToHost`, :py:obj:`~.hipMemcpyDeviceToDevice` or :py:obj:`~.hipMemcpyDefault`.
   dpitch and spitch are the widths in bytes for memory matrix corresponds to dst and src.
   width cannot exceed dpitch or spitch.

   The copy is always performed by the device associated with the specified stream.
   The API is asynchronous with respect to the host, so the call may return before the copy is
   complete. The copy can optionally be excuted in a specific stream by passing a non-zero stream
   argument, for HostToDevice or DeviceToHost copies, the copy can overlap with operations
   in other streams.

   For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is
   attached to the device where the src data is physically located.

   For optimal peer-to-peer copies, the copy device must be able to access the src and dst pointers
   (by calling hipDeviceEnablePeerAccess) with copy agent as the current device and src/dst as the
   peerDevice argument. If enabling device peer access is not done, the API will still work, but
   will perform the copy using a staging buffer on the host.

   Note:
       If host or dst are not pinned, the memory copy will be performed synchronously.  For
       best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpyToArray`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to destination memory address

       dpitch (:py:obj:`~.int`) -- *IN*:
           Pitch size in bytes of destination memory

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to source memory address

       spitch (:py:obj:`~.int`) -- *IN*:
           Pitch size in bytes of source memory

       width (:py:obj:`~.int`) -- *IN*:
           Width of matrix transfer (columns in bytes)

       height (:py:obj:`~.int`) -- *IN*:
           Height of matrix transfer (rows)

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of transfer

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to use

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)

   Copies data between host and device.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpyToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       dst (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       wOffset (:py:obj:`~.int`) -- *IN*:
           Destination starting X offset

       hOffset (:py:obj:`~.int`) -- *IN*:
           Destination starting Y offset

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Source memory address

       spitch (:py:obj:`~.int`) -- *IN*:
           Pitch of source memory

       width (:py:obj:`~.int`) -- *IN*:
           Width of matrix transfer (columns in bytes)

       height (:py:obj:`~.int`) -- *IN*:
           Height of matrix transfer (rows)

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of transfer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind)


.. py:function:: hipMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)

   Copies data between host and device.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpyToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       dst (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       wOffset (:py:obj:`~.int`) -- *IN*:
           Destination starting X offset

       hOffset (:py:obj:`~.int`) -- *IN*:
           Destination starting Y offset

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Source memory address

       spitch (:py:obj:`~.int`) -- *IN*:
           Pitch of source memory

       width (:py:obj:`~.int`) -- *IN*:
           Width of matrix transfer (columns in bytes)

       height (:py:obj:`~.int`) -- *IN*:
           Height of matrix transfer (rows)

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of transfer

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Accelerator view which the copy is being enqueued

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)

   Copies data between host and device.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpyToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       dst (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       wOffsetDst (:py:obj:`~.int`) -- *IN*:
           Destination starting X offset

       hOffsetDst (:py:obj:`~.int`) -- *IN*:
           Destination starting Y offset

       src (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Source memory address

       wOffsetSrc (:py:obj:`~.int`) -- *IN*:
           Source starting X offset

       hOffsetSrc (:py:obj:`~.int`) -- *IN*:
           Source starting Y offset (columns in bytes)

       width (:py:obj:`~.int`) -- *IN*:
           Width of matrix transfer (columns in bytes)

       height (:py:obj:`~.int`) -- *IN*:
           Height of matrix transfer (rows)

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of transfer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, hipMemcpyKind kind)


.. py:function:: hipMemcpyToArray(dst, wOffset, hOffset, src, count, kind)

   Copies data between host and device [Deprecated]

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Warning:
       This API is deprecated.

   Args:
       dst (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       wOffset (:py:obj:`~.int`) -- *IN*:
           Destination starting X offset

       hOffset (:py:obj:`~.int`) -- *IN*:
           Destination starting Y offset

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Source memory address

       count (:py:obj:`~.int`) -- *IN*:
           size in bytes to copy

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of transfer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, hipMemcpyKind kind)


.. py:function:: hipMemcpyFromArray(dst, srcArray, wOffset, hOffset, count, kind)

   Copies data between host and device [Deprecated]

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Warning:
       This API is deprecated.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       srcArray (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Source memory address

       wOffset (:py:obj:`~.int`) -- *IN*:
           Source starting X offset

       hOffset (:py:obj:`~.int`) -- *IN*:
           Source starting Y offset

       count (:py:obj:`~.int`) -- *IN*:
           Size in bytes to copy

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of transfer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyFromArray(void * dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset, size_t count, hipMemcpyKind kind)


.. py:function:: hipMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)

   Copies data between host and device.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       dpitch (:py:obj:`~.int`) -- *IN*:
           Pitch of destination memory

       src (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Source memory address

       wOffset (:py:obj:`~.int`) -- *IN*:
           Source starting X offset

       hOffset (:py:obj:`~.int`) -- *IN*:
           Source starting Y offset

       width (:py:obj:`~.int`) -- *IN*:
           Width of matrix transfer (columns in bytes)

       height (:py:obj:`~.int`) -- *IN*:
           Height of matrix transfer (rows)

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of transfer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DFromArray(void * dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind)


.. py:function:: hipMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)

   Copies data between host and device asynchronously.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       dpitch (:py:obj:`~.int`) -- *IN*:
           Pitch of destination memory

       src (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Source memory address

       wOffset (:py:obj:`~.int`) -- *IN*:
           Source starting X offset

       hOffset (:py:obj:`~.int`) -- *IN*:
           Source starting Y offset

       width (:py:obj:`~.int`) -- *IN*:
           Width of matrix transfer (columns in bytes)

       height (:py:obj:`~.int`) -- *IN*:
           Height of matrix transfer (rows)

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           Type of transfer

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Accelerator view which the copy is being enqueued

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DFromArrayAsync(void * dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpyAtoH(dst, srcArray, srcOffset, count)

   Copies data between host and device.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       srcArray (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Source array

       srcOffset (:py:obj:`~.int`) -- *IN*:
           Offset in bytes of source array

       count (:py:obj:`~.int`) -- *IN*:
           Size of memory copy in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyAtoH(void * dst, hipArray_t srcArray, size_t srcOffset, size_t count)


.. py:function:: hipMemcpyHtoA(dstArray, dstOffset, srcHost, count)

   Copies data between host and device.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       dstArray (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *OUT*:
           Destination memory address

       dstOffset (:py:obj:`~.int`) -- *IN*:
           Offset in bytes of destination array

       srcHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Source host pointer

       count (:py:obj:`~.int`) -- *IN*:
           Size of memory copy in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void * srcHost, size_t count)


.. py:function:: hipMemcpy3D(p)

   Copies data between host and device.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       p (:py:obj:`~.hipMemcpy3DParms`/:py:obj:`~.object`) -- *IN*:
           3D memory copy parameters

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy3D(const struct hipMemcpy3DParms * p)


.. py:function:: hipMemcpy3DAsync(p, stream)

   Copies data between host and device asynchronously.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       p (:py:obj:`~.hipMemcpy3DParms`/:py:obj:`~.object`) -- *IN*:
           3D memory copy parameters

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to use

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms * p, hipStream_t stream)


.. py:function:: hipDrvMemcpy3D(pCopy)

   Copies data between host and device.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       pCopy (:py:obj:`~.HIP_MEMCPY3D`/:py:obj:`~.object`) -- *IN*:
           3D memory copy parameters

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D * pCopy)


.. py:function:: hipDrvMemcpy3DAsync(pCopy, stream)

   Copies data between host and device asynchronously.

   See:
       :py:obj:`~.hipMemcpy`, :py:obj:`~.hipMemcpy2DToArray`, :py:obj:`~.hipMemcpy2D`, :py:obj:`~.hipMemcpyFromArray`, :py:obj:`~.hipMemcpyToSymbol`,
       :py:obj:`~.hipMemcpyAsync`

   Args:
       pCopy (:py:obj:`~.HIP_MEMCPY3D`/:py:obj:`~.object`) -- *IN*:
           3D memory copy parameters

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream to use

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidPitchValue`,
           :py:obj:`~.hipErrorInvalidDevicePointer`, :py:obj:`~.hipErrorInvalidMemcpyDirection`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D * pCopy, hipStream_t stream)


.. py:function:: hipMemGetAddressRange(dptr)

   Get information on memory allocations.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Args:
       dptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Device Pointer

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotFound`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               - BAse pointer address
       * :py:obj:`~.int`:
               - Size of allocation

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemGetAddressRange(hipDeviceptr_t * pbase, size_t * psize, hipDeviceptr_t dptr)


.. py:function:: hipMemcpyBatchAsync(srcs, count, attrs, numAttrs, stream)

   Perform Batch of 1D copies

   Args:
       srcs (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           - Array of source pointers.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of dsts, srcs and sizes arrays

       attrs (:py:obj:`~.hipMemcpyAttributes`/:py:obj:`~.object`) -- *IN*:
           - Array of memcpy attributes (not supported)

       numAttrs (:py:obj:`~.int`) -- *IN*:
           - Size of attrs and attrsIdxs arrays (not supported)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - stream used to enqueue operations in.

   Returns:
       A :py:obj:`~.tuple` of size 5 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               - Array of destination pointers
       * :py:obj:`~.int`:
               - Array of sizes for memcpy operations
       * :py:obj:`~.int`:
               - Array of indices to map attrs to copies (not supported)
       * :py:obj:`~.int`:
               - Pointer to a location to return failure index inside the batch

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyBatchAsync(void ** dsts, void ** srcs, size_t * sizes, size_t count, hipMemcpyAttributes * attrs, size_t * attrsIdxs, size_t numAttrs, size_t * failIdx, hipStream_t stream)


.. py:function:: hipMemcpy3DBatchAsync(numOps, opList, flags, stream)

   Perform Batch of 3D copies

   Args:
       numOps (:py:obj:`~.int`) -- *IN*:
           - Total number of memcpy operations.

       opList (:py:obj:`~.hipMemcpy3DBatchOp`/:py:obj:`~.object`) -- *IN*:
           - Array of size numOps containing the actual memcpy operations.

       flags (:py:obj:`~.int`) -- *IN*:
           - Flags for future use, must be zero now.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - The stream to enqueue the operations in.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               - Pointer to a location to return the index of the copy where a failure
               - was encountered.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy3DBatchAsync(size_t numOps, struct hipMemcpy3DBatchOp * opList, size_t * failIdx, unsigned long long flags, hipStream_t stream)


.. py:function:: hipMemcpy3DPeer(p)

   Performs 3D memory copies between devices
   This API is asynchronous with respect to host

   Args:
       p (:py:obj:`~.hipMemcpy3DPeerParms`/:py:obj:`~.object`) -- *IN*:
           - Parameters for memory copy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, hipErrorInvalidDevice

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy3DPeer(hipMemcpy3DPeerParms * p)


.. py:function:: hipMemcpy3DPeerAsync(p, stream)

   Performs 3D memory copies between devices asynchronously

   Args:
       p (:py:obj:`~.hipMemcpy3DPeerParms`/:py:obj:`~.object`) -- *IN*:
           - Parameters for memory copy

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream to enqueue operation in.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, hipErrorInvalidDevice

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy3DPeerAsync(hipMemcpy3DPeerParms * p, hipStream_t stream)


.. py:function:: hipMipmappedArrayGetMemoryRequirements(mipmap, device)

   Returns the memory requirements of a HIP mipmapped array.

   Returns the memory requirements of a HIP mipmapped array in memoryRequirements.

     The returned value in hipArrayMemoryRequirements::size represents the total size of the HIP
   mipmapped array. The returned value in hipArrayMemoryRequirements::alignment represents the
   alignment necessary for mapping the HIP mipmapped array.

   Args:
       mipmap (:py:obj:`~.hipMipmappedArray`/:py:obj:`~.object`) -- *IN*:
           HIP mipmapped array to get the memory requirements of

       device (:py:obj:`~.int`) -- *IN*:
           Device to get the memory requirements for

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipArrayMemoryRequirements`:
               Pointer to hipArrayMemoryRequirements

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMipmappedArrayGetMemoryRequirements(hipArrayMemoryRequirements * memoryRequirements, hipMipmappedArray_t mipmap, hipDevice_t device)


.. py:function:: hipDeviceCanAccessPeer(deviceId, peerDeviceId)

   Determines if a device can access a peer device's memory.

   *  
   This section describes the PeerToPeer device memory access functions of HIP runtime API.

   The value of ``canAccessPeer,``

   Returns "1" if the specified ``deviceId`` is capable of directly accessing memory physically
   located on ``peerDeviceId,``

   Returns "0" if the specified ``deviceId`` is not capable of directly accessing memory physically
   located on ``peerDeviceId.``

   Returns "0" if ``deviceId`` == ``peerDeviceId,`` both are valid devices,
   however, a device is not a peer of itself.

   Returns :py:obj:`~.hipErrorInvalidDevice` if deviceId or peerDeviceId are not valid devices

   Args:
       deviceId (:py:obj:`~.int`) -- *IN*:
           - The device accessing the peer device memory.

       peerDeviceId (:py:obj:`~.int`) -- *IN*:
           - Peer device where memory is physically located

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.int`:
               - Returns the peer access capability (0 or 1)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceCanAccessPeer(int * canAccessPeer, int deviceId, int peerDeviceId)


.. py:function:: hipDeviceEnablePeerAccess(peerDeviceId, flags)

   Enables direct access to memory allocations on a peer device.

   When this API is successful, all memory allocations on peer device will be mapped into the
   address space of the current device. In addition, any future memory allocation on the
   peer device will remain accessible from the current device, until the access is disabled using
   hipDeviceDisablePeerAccess or device is reset using hipDeviceReset.

   Args:
       peerDeviceId (:py:obj:`~.int`) -- *IN*:
           - Peer device to enable direct access to from the current device

       flags (:py:obj:`~.int`) -- *IN*:
           - Reserved for future use, must be zero

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: One of:
               - py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`,
               - py:obj:`~.hipErrorPeerAccessAlreadyEnabled` if peer access is already enabled for this device.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags)


.. py:function:: hipDeviceDisablePeerAccess(peerDeviceId)

   Disables direct access to memory allocations on a peer device.

   If direct access to memory allocations on peer device has not been enabled yet from the current
   device, it returns :py:obj:`~.hipErrorPeerAccessNotEnabled`.

   Args:
       peerDeviceId (:py:obj:`~.int`) -- *IN*:
           Peer device to disable direct access to

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorPeerAccessNotEnabled`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceDisablePeerAccess(int peerDeviceId)


.. py:function:: hipMemcpyPeer(dst, dstDeviceId, src, srcDeviceId, sizeBytes)

   Copies memory between two peer accessible devices.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           - Destination device pointer

       dstDeviceId (:py:obj:`~.int`) -- *IN*:
           - Destination device

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Source device pointer

       srcDeviceId (:py:obj:`~.int`) -- *IN*:
           - Source device

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           - Size of memory copy in bytes

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDevice`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyPeer(void * dst, int dstDeviceId, const void * src, int srcDeviceId, size_t sizeBytes)


.. py:function:: hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDevice, sizeBytes, stream)

   Copies memory between two peer accessible devices asynchronously.

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           - Destination device pointer

       dstDeviceId (:py:obj:`~.int`) -- *IN*:
           - Destination device

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Source device pointer

       srcDevice (:py:obj:`~.int`) -- *IN*:
           - Source device

       sizeBytes (:py:obj:`~.int`) -- *IN*:
           - Size of memory copy in bytes

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream identifier

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDevice`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyPeerAsync(void * dst, int dstDeviceId, const void * src, int srcDevice, size_t sizeBytes, hipStream_t stream)


.. py:function:: hipDeviceGetDevResource(device, type)

   Gets device resource of a given type for a device.

   *  This section describes execution context management functions of HIP runtime API.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           - Device to get resource for

       type (:py:obj:`~.hipDevResourceType`) -- *IN*:
           - Type of resource to retrieve

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceType`,
           :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.hipDevResource_st`:
               - Output device resource pointer

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetDevResource(hipDevice_t device, hipDevResource * resource, hipDevResourceType type)


.. py:function:: hipDevSmResourceSplitByCount(input, remainder, flags, minCount)

   Splits SM resources into groups containing the specified number of SMs.

   Args:
       input (:py:obj:`~.hipDevResource_st`/:py:obj:`~.object`) -- *IN*:
           - Valid input SM resource to be split

       remainder (:py:obj:`~.hipDevResource_st`/:py:obj:`~.object`) -- *IN*:
           - If the input resource cannot be evenly split among nbGroups,
           the remaining resourced are returned through this parameter.

       flags (:py:obj:`~.int`) -- *IN*:
           - Flags specifying partition usage and constraints to apply when splitting
           the inout resource.

       minCount (:py:obj:`~.int`) -- *IN*:
           - Specifies the minimum number of SMs required

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceType`,
           :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipDevResource_st`:
               - Output device resource pointer
       * :py:obj:`~.int`:
               - The poiter specifying the number of groups

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDevSmResourceSplitByCount(hipDevResource * result, unsigned int * nbGroups, const hipDevResource * input, hipDevResource * remainder, unsigned int flags, unsigned int minCount)


.. py:function:: hipDevSmResourceSplit(nbGroups, input, remainder, flags, groupParams)

   Splits SM resources into structured groups.

   Args:
       nbGroups (:py:obj:`~.int`) -- *IN*:
           - The poiter specifying the number of groups

       input (:py:obj:`~.hipDevResource_st`/:py:obj:`~.object`) -- *IN*:
           - Valid input SM resource to be split

       remainder (:py:obj:`~.hipDevResource_st`/:py:obj:`~.object`) -- *IN*:
           - If the input resource cannot be evenly split among nbGroups,
           the remaining resourced are returned through this parameter.

       flags (:py:obj:`~.int`) -- *IN*:
           - Flags specifying partition usage and constraints to apply when splitting
           the inout resource.

       groupParams (:py:obj:`~.hipDevSmResourceGroupParams_st`/:py:obj:`~.object`) -- *IN*:
           - Describes how the SM resources should be partitioned and assigned
           to the corresponding result entries.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceType`,
           :py:obj:`~.hipErrorInvalidResourceConfiguration`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.hipDevResource_st`:
               - Output device resource pointer

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDevSmResourceSplit(hipDevResource * result, unsigned int nbGroups, const hipDevResource * input, hipDevResource * remainder, unsigned int flags, hipDevSmResourceGroupParams * groupParams)


.. py:function:: hipDevResourceGenerateDesc(resources, nbResources)

   Generates a resource descriptor from one or more device resources.

   Args:
       resources (:py:obj:`~.hipDevResource_st`/:py:obj:`~.object`) -- *IN*:
           - Pointer of device resources to be included in the descriptor

       nbResources (:py:obj:`~.int`) -- *IN*:
           - Number of resources specified

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceType`,
           :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.ihipDevResourceDesc_t`:
               - Output parameter that receives the generated resource descriptor

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDevResourceGenerateDesc(hipDevResourceDesc_t * phDesc, hipDevResource * resources, unsigned int nbResources)


.. py:function:: hipGreenCtxCreate(desc, device, flags)

   Creates a green context from a resource descriptor.

   Args:
       desc (:py:obj:`~.ihipDevResourceDesc_t`/:py:obj:`~.object`) -- *IN*:
           - Resource descriptor generated via hipDevResourceGenerateDesc that specifies
           the set of resources to be used

       device (:py:obj:`~.int`) -- *IN*:
           - Device on which the green context is created

       flags (:py:obj:`~.int`) -- *IN*:
           - Flags controlling green context creation

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDevice`
       * :py:obj:`~.ihipExecutionCtx_t`:
               - Output parameter that receives the handle to the created green context

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGreenCtxCreate(hipExecutionCtx_t * ctx, hipDevResourceDesc_t desc, int device, unsigned int flags)


.. py:function:: hipExecutionCtxDestroy(ctx)

   Destroys an execution context.

   Args:
       ctx (:py:obj:`~.ihipExecutionCtx_t`/:py:obj:`~.object`) -- *IN*:
           - Execution context to destroy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExecutionCtxDestroy(hipExecutionCtx_t ctx)


.. py:function:: hipDeviceGetExecutionCtx(device)

   Returns the default execution context for a device.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           - The device on which to receive the execution context

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.ihipExecutionCtx_t`:
               - Output pointer for execution context

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetExecutionCtx(hipExecutionCtx_t * ctx, int device)


.. py:function:: hipExecutionCtxStreamCreate(greenctx, flags, priority)

   Creates a stream on an execution context with specified flags and priority

   Args:
       greenctx (:py:obj:`~.ihipExecutionCtx_t`/:py:obj:`~.object`) -- *IN*:
           - Execution context used to create and initialize the stream

       flags (:py:obj:`~.int`) -- *IN*:
           - Flags for stream creation

       priority (:py:obj:`~.int`) -- *IN*:
           - Stream priority

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.ihipStream_t`:
               - Output pointer of the created stream

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExecutionCtxStreamCreate(hipStream_t * stream, hipExecutionCtx_t greenctx, unsigned int flags, int priority)


.. py:function:: hipExecutionCtxGetDevResource(ctx, type)

   Returns the device resource of a given type for an execution context

   Args:
       ctx (:py:obj:`~.ihipExecutionCtx_t`/:py:obj:`~.object`) -- *IN*:
           - Execution context to get resource for

       type (:py:obj:`~.hipDevResourceType`) -- *IN*:
           - Type of device resource

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipDevResource_st`:
               - Output pointer that receives the structured device resource

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExecutionCtxGetDevResource(hipExecutionCtx_t ctx, hipDevResource * resource, hipDevResourceType type)


.. py:function:: hipExecutionCtxGetDevice(ctx)

   Returns the device associated with an execution context

   Args:
       ctx (:py:obj:`~.ihipExecutionCtx_t`/:py:obj:`~.object`) -- *IN*:
           - Execution context to obtain the device

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               - Returns device handle for the specified execution context

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExecutionCtxGetDevice(int * device, hipExecutionCtx_t ctx)


.. py:function:: hipExecutionCtxGetId(ctx)

   Returns a unique identifier for an execution context

   Args:
       ctx (:py:obj:`~.ihipExecutionCtx_t`/:py:obj:`~.object`) -- *IN*:
           - Execution context to obtain the ID

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               - Pointer to the context ID

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExecutionCtxGetId(hipExecutionCtx_t ctx, unsigned long long * ctxId)


.. py:function:: hipStreamGetDevResource(hStream, type)

   Returns the device resource of a given type for a stream

   Args:
       hStream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream to get resource for

       type (:py:obj:`~.hipDevResourceType`) -- *IN*:
           - Type of resource

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceType`, :py:obj:`~.hipErrorInvalidHandle`
       * :py:obj:`~.hipDevResource_st`:
               - Pointer to the structured device resource

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetDevResource(hipStream_t hStream, hipDevResource * resource, hipDevResourceType type)


.. py:function:: hipExecutionCtxRecordEvent(ctx)

   Records an event on an execution context

   Args:
       ctx (:py:obj:`~.ihipExecutionCtx_t`/:py:obj:`~.object`) -- *IN*:
           - Execution context to record event for

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidHandle`
       * :py:obj:`~.ihipEvent_t`:
               - Event to record

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExecutionCtxRecordEvent(hipExecutionCtx_t ctx, hipEvent_t event)


.. py:function:: hipExecutionCtxSynchronize(ctx)

   Blocks until all work on an execution context has completed

   Args:
       ctx (:py:obj:`~.ihipExecutionCtx_t`/:py:obj:`~.object`) -- *IN*:
           - Execution context to synchronize

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExecutionCtxSynchronize(hipExecutionCtx_t ctx)


.. py:function:: hipExecutionCtxWaitEvent(ctx, event)

   Makes an execution context wait on an event

   Args:
       ctx (:py:obj:`~.ihipExecutionCtx_t`/:py:obj:`~.object`) -- *IN*:
           - Execution context to wait for

       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           - Event to wait on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExecutionCtxWaitEvent(hipExecutionCtx_t ctx, hipEvent_t event)


.. py:function:: hipCtxCreate(flags, device)

   Create a context and set it as current/default context

   See:
       :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`, :py:obj:`~.hipCtxPushCurrent`,
       :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Args:
       flags (:py:obj:`~.int`) -- *IN*:
           Context creation flags

       device (:py:obj:`~.int`) -- *IN*:
           device handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`
       * :py:obj:`~.ihipCtx_t`:
               Context to create

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxCreate(hipCtx_t * ctx, unsigned int flags, hipDevice_t device)


.. py:function:: hipCtxDestroy(ctx)

   Destroy a HIP context [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,:py:obj:`~.hipCtxSetCurrent`,
       :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize` , :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Args:
       ctx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           Context to destroy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxDestroy(hipCtx_t ctx)


.. py:function:: hipCtxPopCurrent()

   Pop the current/default context and return the popped context [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidContext`
       * :py:obj:`~.ihipCtx_t`:
               The current context to pop

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxPopCurrent(hipCtx_t * ctx)


.. py:function:: hipCtxPushCurrent(ctx)

   Push the context to be set as current/ default context [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize` , :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Args:
       ctx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           The current context to push

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidContext`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxPushCurrent(hipCtx_t ctx)


.. py:function:: hipCtxSetCurrent(ctx)

   Set the passed context as current/default [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize` , :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Args:
       ctx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           The context to set as current

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidContext`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxSetCurrent(hipCtx_t ctx)


.. py:function:: hipCtxGetCurrent()

   Get the handle of the current/ default context [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetDevice`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`,
       :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidContext`
       * :py:obj:`~.ihipCtx_t`:
               The context to get as current

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxGetCurrent(hipCtx_t * ctx)


.. py:function:: hipCtxGetDevice()

   Get the handle of the device associated with current/default context [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidContext`
       * :py:obj:`~.int`:
               The device from the current context

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxGetDevice(hipDevice_t * device)


.. py:function:: hipCtxGetApiVersion(ctx)

   Returns the approximate HIP api version.

   Warning:
       The HIP feature set does not correspond to an exact CUDA SDK api revision.
       This function always set *apiVersion to 4 as an approximation though HIP supports
       some features which were introduced in later CUDA SDK revisions.
       HIP apps code should not rely on the api revision number here and should
       use arch feature flags to test device capabilities or conditional compilation.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetDevice`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`,
       :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Args:
       ctx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           Context to check [Deprecated]

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`
       * :py:obj:`~.int`:
               API version to get

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxGetApiVersion(hipCtx_t ctx, unsigned int * apiVersion)


.. py:function:: hipCtxGetCacheConfig()

   Get Cache configuration for a specific function [Deprecated]

   Warning:
       AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
       ignored on those architectures.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`
       * :py:obj:`~.hipFuncCache_t`:
               Cache configuration

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxGetCacheConfig(hipFuncCache_t * cacheConfig)


.. py:function:: hipCtxSetCacheConfig(cacheConfig)

   Set L1/Shared cache partition [Deprecated]

   Warning:
       AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
       ignored on those architectures.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Args:
       cacheConfig (:py:obj:`~.hipFuncCache_t`) -- *IN*:
           Cache configuration to set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig)


.. py:function:: hipCtxSetSharedMemConfig(config)

   Set Shared memory bank configuration  [Deprecated]

   Warning:
       AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
       ignored on those architectures.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Args:
       config (:py:obj:`~.hipSharedMemConfig`) -- *IN*:
           Shared memory configuration to set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config)


.. py:function:: hipCtxGetSharedMemConfig()

   Get Shared memory bank configuration [Deprecated]

   Warning:
       AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
       ignored on those architectures.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`
       * :py:obj:`~.hipSharedMemConfig`:
               Pointer of shared memory configuration

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig * pConfig)


.. py:function:: hipCtxSynchronize()

   Blocks until the default context has completed all preceding requested tasks [Deprecated]

   Warning:
       This function waits for all streams on the default context to complete execution, and
       then returns.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxSynchronize()


.. py:function:: hipCtxGetFlags()

   Return flags used for creating default context [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`
       * :py:obj:`~.int`:
               Pointer of flags

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxGetFlags(unsigned int * flags)


.. py:function:: hipCtxEnablePeerAccess(peerCtx, flags)

   Enables direct access to memory allocations in a peer context [Deprecated]

   Memory which already allocated on peer device will be mapped into the address space of the
   current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
   the address space of the current device when the memory is allocated. The peer memory remains
   accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       PeerToPeer support is experimental.

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Args:
       peerCtx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           Peer context

       flags (:py:obj:`~.int`) -- *IN*:
           flags, need to set as 0

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorPeerAccessAlreadyEnabled`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags)


.. py:function:: hipCtxDisablePeerAccess(peerCtx)

   Disable direct access from current context's virtual address space to memory allocations
   physically located on a peer context.Disables direct access to memory allocations in a peer
   context and unregisters any registered allocations [Deprecated]

   Returns :py:obj:`~.hipErrorPeerAccessNotEnabled` if direct access to memory on peerDevice has not yet been
   enabled from the current device.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       PeerToPeer support is experimental.

   Warning:
       This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
       NVIDIA platform.

   Args:
       peerCtx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           Peer context to be disabled

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorPeerAccessNotEnabled`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx)


.. py:function:: hipDevicePrimaryCtxGetState(dev)

   Get the state of the primary context [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent driver API on the
       NVIDIA platform.

   Args:
       dev (:py:obj:`~.int`) -- *IN*:
           Device to get primary context flags for

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`
       * :py:obj:`~.int`:
               Pointer to store flags
       * :py:obj:`~.int`:
               Pointer to store context state; 0 = inactive, 1 = active

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int * flags, int * active)


.. py:function:: hipDevicePrimaryCtxRelease(dev)

   Release the primary context on the GPU.

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This function return :py:obj:`~.hipSuccess` though doesn't release the primaryCtx by design on
       HIP/HIP-CLANG path.

   Warning:
       This API is deprecated on the AMD platform, only for equivalent driver API on the
       NVIDIA platform.

   Args:
       dev (:py:obj:`~.int`) -- *IN*:
           Device which primary context is released [Deprecated]

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev)


.. py:function:: hipDevicePrimaryCtxRetain(dev)

   Retain the primary context on the GPU [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent driver API on the
       NVIDIA platform.

   Args:
       dev (:py:obj:`~.int`) -- *IN*:
           Device which primary context is released

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`
       * :py:obj:`~.ihipCtx_t`:
               Returned context handle of the new context

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDevicePrimaryCtxRetain(hipCtx_t * pctx, hipDevice_t dev)


.. py:function:: hipDevicePrimaryCtxReset(dev)

   Resets the primary context on the GPU [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent driver API on the
       NVIDIA platform.

   Args:
       dev (:py:obj:`~.int`) -- *IN*:
           Device which primary context is reset

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev)


.. py:function:: hipDevicePrimaryCtxSetFlags(dev, flags)

   Set flags for the primary context [Deprecated]

   See:
       :py:obj:`~.hipCtxCreate`, :py:obj:`~.hipCtxDestroy`, :py:obj:`~.hipCtxGetFlags`, :py:obj:`~.hipCtxPopCurrent`, :py:obj:`~.hipCtxGetCurrent`,
       :py:obj:`~.hipCtxSetCurrent`, :py:obj:`~.hipCtxPushCurrent`, :py:obj:`~.hipCtxSetCacheConfig`, :py:obj:`~.hipCtxSynchronize`, :py:obj:`~.hipCtxGetDevice`

   Warning:
       This API is deprecated on the AMD platform, only for equivalent driver API on the
       NVIDIA platform.

   Args:
       dev (:py:obj:`~.int`) -- *IN*:
           Device for which the primary context flags are set

       flags (:py:obj:`~.int`) -- *IN*:
           New flags for the device

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorContextAlreadyInUse`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags)


.. py:function:: hipModuleLoadFatBinary(fatbin)

   Loads fatbin object

   *  
   This section describes the module management functions of HIP runtime API.

   Args:
       fatbin (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           fatbin to be loaded as a module

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidContext`, :py:obj:`~.hipErrorFileNotFound`,
           :py:obj:`~.hipErrorOutOfMemory`, :py:obj:`~.hipErrorSharedObjectInitFailed`, :py:obj:`~.hipErrorNotInitialized`
       * :py:obj:`~.ihipModule_t`:
               Module

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleLoadFatBinary(hipModule_t * module, const void * fatbin)


.. py:function:: hipModuleLoad(fname)

   Loads code object from file into a module the currrent context.

   Warning:
       File/memory resources allocated in this function are released only in hipModuleUnload.

   Args:
       fname (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Filename of code object to load

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidContext`, :py:obj:`~.hipErrorFileNotFound`,
           :py:obj:`~.hipErrorOutOfMemory`, :py:obj:`~.hipErrorSharedObjectInitFailed`, :py:obj:`~.hipErrorNotInitialized`
       * :py:obj:`~.ihipModule_t`:
               Module

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleLoad(hipModule_t * module, const char * fname)


.. py:function:: hipModuleUnload(module)

   Frees the module

   The module is freed, and the code objects associated with it are destroyed.

   Args:
       module (:py:obj:`~.ihipModule_t`/:py:obj:`~.object`) -- *IN*:
           Module to free

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidResourceHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleUnload(hipModule_t module)


.. py:function:: hipModuleGetFunction(module, kname)

   Function with kname will be extracted if present in module

   Args:
       module (:py:obj:`~.ihipModule_t`/:py:obj:`~.object`) -- *IN*:
           Module to get function from

       kname (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Pointer to the name of function

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidContext`, :py:obj:`~.hipErrorNotInitialized`,
           :py:obj:`~.hipErrorNotFound`,
       * :py:obj:`~.ihipModuleSymbol_t`:
               Pointer to function handle

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleGetFunction(hipFunction_t * function, hipModule_t module, const char * kname)


.. py:function:: hipModuleGetFunctionCount(mod)

   Returns the number of functions within a module.

   Args:
       mod (:py:obj:`~.ihipModule_t`/:py:obj:`~.object`) -- *IN*:
           Module to get function count from

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidContext`, :py:obj:`~.hipErrorNotInitialized`,
           :py:obj:`~.hipErrorNotFound`,
       * :py:obj:`~.int`:
               function count from module

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleGetFunctionCount(unsigned int * count, hipModule_t mod)


.. py:function:: hipKernelGetAttribute(attrib, kernel, dev)

   Returns information about a kernel.

   Returns in *pi the integer value of the attribute attrib for the kernel kernel for the requested
   device dev. The supported attributes are:
     - HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK The maximum number of threads per block. This number depends on both the kernel and the requested device.
     - HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES The size in bytes of statically-allocated shared memory per block required by this kernel. This does not include dynamically-allocated shared memory requested by the user at runtime.
     - HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES The size in bytes of user-allocated constant memory required by this kernel.
     - HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES The size in bytes of local memory used by each thread of this kernel.
     - HIP_FUNC_ATTRIBUTE_NUM_REGS The number of registers used by each thread of this kernel.
     - HIP_FUNC_ATTRIBUTE_PTX_VERSION The PTX virtual architecture version for which the kernel was compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13.
     - HIP_FUNC_ATTRIBUTE_BINARY_VERSION The binary architecture version for which the kernel was compiled. This value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return the value 13.
     - HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES The maximum size in bytes of dynamically-allocated shared memory.
     - HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA The attribute to indicate whether the kernel has been compiled with user specified option "-Xptxas --dlcm=ca" set.
     - HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT Preferred shared memory-L1 cache split ratio in percent of total shared memory.

   See:
       :py:obj:`~.hipLibraryLoadData`, :py:obj:`~.hipLibraryLoadFromFile`, :py:obj:`~.hipLibraryUnload`, :py:obj:`~.hipKernelSetAttribute`,
       :py:obj:`~.hipLibraryGetKernel`, :py:obj:`~.hipLaunchKernel`, :py:obj:`~.hipKernelGetFunction`, :py:obj:`~.hipLibraryGetModule`,
       :py:obj:`~.hipModuleGetFunction`, :py:obj:`~.hipFuncGetAttribute`

   Args:
       attrib (:py:obj:`~.hipFunction_attribute`) -- *IN*:
           Attribute requested

       kernel (:py:obj:`~.ihipKernel_t`/:py:obj:`~.object`) -- *IN*:
           Kernel to query attribute of

       dev (:py:obj:`~.int`) -- *IN*:
           Device to query attribute of

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidDeviceFunction`, :py:obj:`~.hipErrorMissingConfiguration`
       * :py:obj:`~.int`:
               Returned attribute value

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipKernelGetAttribute(int * pi, hipFunction_attribute attrib, hipKernel_t kernel, hipDevice_t dev)


.. py:function:: hipLibraryLoadData(code, jitOptionsValues, numJitOptions, libraryOptionValues, numLibraryOptions)

   Load hip Library from inmemory object

   Args:
       code (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           In memory object

       jitOptionsValues (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           JIT options values, CUDA only

       numJitOptions (:py:obj:`~.int`) -- *IN*:
           Number of JIT options

       libraryOptionValues (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Library options values

       numLibraryOptions (:py:obj:`~.int`) -- *IN*:
           Number of library options

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`,
       * :py:obj:`~.ihipLibrary_t`:
               Output Library
       * :py:obj:`~.hipJitOption`:
               JIT options, CUDA only
       * :py:obj:`~.hipLibraryOption_e`:
               Library options

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLibraryLoadData(hipLibrary_t * library, const void * code, hipJitOption * jitOptions, void ** jitOptionsValues, unsigned int numJitOptions, hipLibraryOption * libraryOptions, void ** libraryOptionValues, unsigned int numLibraryOptions)


.. py:function:: hipLibraryLoadFromFile(fileName, jitOptionsValues, numJitOptions, libraryOptionValues, numLibraryOptions)

   Load hip Library from file

   Args:
       fileName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           file which contains code object

       jitOptionsValues (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           JIT options values, CUDA only

       numJitOptions (:py:obj:`~.int`) -- *IN*:
           Number of JIT options

       libraryOptionValues (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Library options values

       numLibraryOptions (:py:obj:`~.int`) -- *IN*:
           Number of library options

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipLibrary_t`:
               Output Library
       * :py:obj:`~.hipJitOption`:
               JIT options, CUDA only
       * :py:obj:`~.hipLibraryOption_e`:
               Library options

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLibraryLoadFromFile(hipLibrary_t * library, const char * fileName, hipJitOption * jitOptions, void ** jitOptionsValues, unsigned int numJitOptions, hipLibraryOption * libraryOptions, void ** libraryOptionValues, unsigned int numLibraryOptions)


.. py:function:: hipLibraryUnload(library)

   Unload HIP Library

   Args:
       library (:py:obj:`~.ihipLibrary_t`/:py:obj:`~.object`) -- *IN*:
           Input created hip library

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLibraryUnload(hipLibrary_t library)


.. py:function:: hipLibraryGetKernel(library, name)

   Get Kernel object from library

   Args:
       library (:py:obj:`~.ihipLibrary_t`/:py:obj:`~.object`) -- *IN*:
           Input hip library

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           kernel name to be searched for

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipKernel_t`:
               Output kernel object

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLibraryGetKernel(hipKernel_t * pKernel, hipLibrary_t library, const char * name)


.. py:function:: hipLibraryGetKernelCount(library)

   Get Kernel count in library

   Args:
       library (:py:obj:`~.ihipLibrary_t`/:py:obj:`~.object`) -- *IN*:
           Input created hip library

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Count of kernels in library

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLibraryGetKernelCount(unsigned int * count, hipLibrary_t library)


.. py:function:: hipLibraryGetGlobal(library, name)

   Get device pointer to a `__device__` global variable defined in a library.

   Returns the device pointer and size of the named global symbol within the
   library's code object. Mirrors CUDA's `cuLibraryGetGlobal` /
   `cudaLibraryGetGlobal`. Either `dptr` or `bytes` (but not both) may be NULL.

   Args:
       library (:py:obj:`~.ihipLibrary_t`/:py:obj:`~.object`) -- *IN*:
           Input hip library handle.

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Name of the global symbol to look up.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceHandle`,
           :py:obj:`~.hipErrorNotFound`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to receive the device pointer, may be NULL.
       * :py:obj:`~.int`:
               Pointer to receive the size in bytes, may be NULL.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLibraryGetGlobal(void ** dptr, size_t * bytes, hipLibrary_t library, const char * name)


.. py:function:: hipLibraryGetManaged(library, name)

   Get host pointer to a `__managed__` variable defined in a library.

   Returns the host-accessible managed pointer and size of the named managed
   symbol within the library's code object. Mirrors CUDA's
   `cuLibraryGetManaged` / `cudaLibraryGetManaged`. Either `dptr` or `bytes`
   (but not both) may be NULL. Returns :py:obj:`~.hipErrorNotFound` if the symbol does
   not exist or is not a `__managed__` variable.

   Args:
       library (:py:obj:`~.ihipLibrary_t`/:py:obj:`~.object`) -- *IN*:
           Input hip library handle.

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Name of the managed symbol to look up.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidResourceHandle`,
           :py:obj:`~.hipErrorNotFound`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to receive the managed host pointer, may be NULL.
       * :py:obj:`~.int`:
               Pointer to receive the size in bytes, may be NULL.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLibraryGetManaged(void ** dptr, size_t * bytes, hipLibrary_t library, const char * name)


.. py:function:: hipLibraryEnumerateKernels(numKernels, library)

   Retrieve kernel handles within a library

   Args:
       numKernels (:py:obj:`~.int`) -- *IN*:
           Maximum number of kernel handles to return to buffer
           @oaram [in] library Library handle to query from

       library (:py:obj:`~.ihipLibrary_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipKernel_t`:
               Buffer for kernel handles

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLibraryEnumerateKernels(hipKernel_t * kernels, unsigned int numKernels, hipLibrary_t library)


.. py:function:: hipKernelGetLibrary(kernel)

   Returns a Library Handle

   Args:
       kernel (:py:obj:`~.ihipKernel_t`/:py:obj:`~.object`) -- *IN*:
           Kernel to retrieve library Handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipLibrary_t`:
               Returned Library handle

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipKernelGetLibrary(hipLibrary_t * library, hipKernel_t kernel)


.. py:function:: hipKernelGetName(kernel)

   Returns a Kernel Name

   Args:
       kernel (:py:obj:`~.ihipKernel_t`/:py:obj:`~.object`) -- *IN*:
           Kernel handle to retrieve name

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               Returned Kernel Name

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipKernelGetName(const char ** name, hipKernel_t kernel)


.. py:function:: hipKernelGetParamInfo(kernel, paramIndex)

   Returns the offset and size of a kernel parameter

   Args:
       kernel (:py:obj:`~.ihipKernel_t`/:py:obj:`~.object`) -- *IN*:
           Kernel handle to retrieve parameter info

       paramIndex (:py:obj:`~.int`) -- *IN*:
           Index of the parameter

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               returns the offset of the parameter
       * :py:obj:`~.int`:
               Optionally returns the size of the parameter

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipKernelGetParamInfo(hipKernel_t kernel, size_t paramIndex, size_t * paramOffset, size_t * paramSize)


.. py:function:: hipFuncGetAttributes(func)

   Find out attributes for a given function.

   Args:
       func (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the function handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDeviceFunction`
       * :py:obj:`~.hipFuncAttributes`:
               Attributes of funtion

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFuncGetAttributes(struct hipFuncAttributes * attr, const void * func)


.. py:function:: hipFuncGetAttribute(attrib, hfunc)

   Find out a specific attribute for a given function.

   Args:
       attrib (:py:obj:`~.hipFunction_attribute`) -- *IN*:
           Attributes of the given funtion

       hfunc (:py:obj:`~.ihipModuleSymbol_t`/:py:obj:`~.object`) -- *IN*:
           Function to get attributes from

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDeviceFunction`
       * :py:obj:`~.int`:
               Pointer to the value

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFuncGetAttribute(int * value, hipFunction_attribute attrib, hipFunction_t hfunc)


.. py:function:: hipGetFuncBySymbol(symbolPtr)

   Gets pointer to device entry function that matches entry function symbolPtr.

   Args:
       symbolPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to device entry function to search for

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDeviceFunction`
       * :py:obj:`~.ihipModuleSymbol_t`:
               Device entry function

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetFuncBySymbol(hipFunction_t * functionPtr, const void * symbolPtr)


.. py:function:: hipGetDriverEntryPoint(symbol, flags)

   Gets function pointer of a requested HIP API

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           The API base name

       flags (:py:obj:`~.int`) -- *IN*:
           Flags for the search

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to the requested function
       * :py:obj:`~.hipDriverEntryPointQueryResult`:
               Optional returned status of the search

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long flags, hipDriverEntryPointQueryResult * driverStatus)


.. py:function:: hipModuleGetTexRef(hmod, name)

   returns the handle of the texture reference with the name from the module.

   Args:
       hmod (:py:obj:`~.ihipModule_t`/:py:obj:`~.object`) -- *IN*:
           Module

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Pointer of name of texture reference

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorNotFound`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.textureReference`:
               Pointer of texture reference

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleGetTexRef(textureReference ** texRef, hipModule_t hmod, const char * name)


.. py:function:: hipModuleLoadData(image)

   builds module from code object data which resides in host memory.

   The "image" is a pointer to the location of code object data. This data can be either
   a single code object or a fat binary (fatbin), which serves as the entry point for loading and
   launching device-specific kernel executions.

   By default, the following command generates a fatbin:

   "amdclang++ -O3 -c --offload-device-only --offload-arch=<GPU_ARCH> <input_file> -o <output_file>"

   For more details, refer to:
   <a
   href= "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/kernel_language_cpp_support.html:py:obj:`~.kernel`-compilation">
   Kernel Compilation</a> in the HIP kernel language C++ support, or
   <a
   href="https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html">HIP runtime compilation (HIP RTC)</a>.

   Args:
       image (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The pointer to the location of data

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
       * :py:obj:`~.ihipModule_t`:
               Retuned module

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleLoadData(hipModule_t * module, const void * image)


.. py:function:: hipModuleLoadDataEx(image, numOptions, optionValues)

   builds module from code object which resides in host memory. Image is pointer to that
   location. Options are not used. hipModuleLoadData is called.

   Args:
       image (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The pointer to the location of data

       numOptions (:py:obj:`~.int`) -- *IN*:
           Number of options

       optionValues (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Option values for JIT

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
       * :py:obj:`~.ihipModule_t`:
               Retuned module
       * :py:obj:`~.hipJitOption`:
               Options for JIT

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleLoadDataEx(hipModule_t * module, const void * image, unsigned int numOptions, hipJitOption * options, void ** optionValues)


.. py:function:: hipLinkAddData(state, type, data, size, name, numOptions, optionValues)

   Adds bitcode data to be linked with options.

   If adding the file fails, it will

   See:
       :py:obj:`~.hipError_t`

   Args:
       state (:py:obj:`~.ihipLinkState_t`/:py:obj:`~.object`) -- *IN*:
           hip link state

       type (:py:obj:`~.hipJitInputType`) -- *IN*:
           Type of the input data or bitcode

       data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data which is null terminated

       size (:py:obj:`~.int`) -- *IN*:
           Size of the input data

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Optional name for this input

       numOptions (:py:obj:`~.int`) -- *IN*:
           Size of the options

       optionValues (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of option values cast to void*

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: One of:
               - py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidHandle`
               - py:obj:`~.hipErrorInvalidConfiguration`
       * :py:obj:`~.hipJitOption`:
               Array of options applied to this input

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLinkAddData(hipLinkState_t state, hipJitInputType type, void * data, size_t size, const char * name, unsigned int numOptions, hipJitOption * options, void ** optionValues)


.. py:function:: hipLinkAddFile(state, type, path, numOptions, optionValues)

   Adds a file with bitcode to be linked with options.

   If adding the file fails, it will

   See:
       :py:obj:`~.hipError_t`

   Args:
       state (:py:obj:`~.ihipLinkState_t`/:py:obj:`~.object`) -- *IN*:
           hip link state

       type (:py:obj:`~.hipJitInputType`) -- *IN*:
           Type of the input data or bitcode

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Path to the input file where bitcode is present

       numOptions (:py:obj:`~.int`) -- *IN*:
           Size of the options

       optionValues (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of option values cast to void*

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: One of:
               - py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
               - py:obj:`~.hipErrorInvalidConfiguration`
       * :py:obj:`~.hipJitOption`:
               Array of options applied to this input

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLinkAddFile(hipLinkState_t state, hipJitInputType type, const char * path, unsigned int numOptions, hipJitOption * options, void ** optionValues)


.. py:function:: hipLinkComplete(state)

   Completes the linking of the given program.

   If adding the data fails, it will

   See:
       :py:obj:`~.hipError_t`

   Args:
       state (:py:obj:`~.ihipLinkState_t`/:py:obj:`~.object`) -- *IN*:
           hip link state

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: One of:
               - py:obj:`~.hipSuccess` :py:obj:`~.hipErrorInvalidValue`
               - py:obj:`~.hipErrorInvalidConfiguration`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Upon success, points to the output binary
       * :py:obj:`~.int`:
               Size of the binary is stored (optional)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLinkComplete(hipLinkState_t state, void ** hipBinOut, size_t * sizeOut)


.. py:function:: hipLinkCreate(numOptions, optionValues)

   Creates a linker instance with options.

   See:
       :py:obj:`~.hipSuccess`

   Args:
       numOptions (:py:obj:`~.int`) -- *IN*:
           Number of options

       optionValues (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of option values cast to void*

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess` :py:obj:`~.hipErrorInvalidValue` :py:obj:`~.hipErrorInvalidConfiguration`
       * :py:obj:`~.hipJitOption`:
               Array of options
       * :py:obj:`~.ihipLinkState_t`:
               hip link state created upon success

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLinkCreate(unsigned int numOptions, hipJitOption * options, void ** optionValues, hipLinkState_t * stateOut)


.. py:function:: hipLinkDestroy(state)

   Deletes the linker instance.

   See:
       :py:obj:`~.hipSuccess`

   Args:
       state (:py:obj:`~.ihipLinkState_t`/:py:obj:`~.object`) -- *IN*:
           link state instance

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess` :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLinkDestroy(hipLinkState_t state)


.. py:function:: hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra)

   launches kernel f with launch parameters and shared memory on stream with arguments passed
   to kernelparams or extra

   Please note, HIP does not support kernel launch with total work items defined in dimension with
   size gridDim x blockDim >= 2^32. So gridDim.x * blockDim.x, gridDim.y * blockDim.y
   and gridDim.z * blockDim.z are always less than 2^32.

   Args:
       f (:py:obj:`~.ihipModuleSymbol_t`/:py:obj:`~.object`) -- *IN*:
           Kernel to launch.

       gridDimX (:py:obj:`~.int`) -- *IN*:
           X grid dimension specified as multiple of blockDimX.

       gridDimY (:py:obj:`~.int`) -- *IN*:
           Y grid dimension specified as multiple of blockDimY.

       gridDimZ (:py:obj:`~.int`) -- *IN*:
           Z grid dimension specified as multiple of blockDimZ.

       blockDimX (:py:obj:`~.int`) -- *IN*:
           X block dimensions specified in work-items

       blockDimY (:py:obj:`~.int`) -- *IN*:
           Y grid dimension specified in work-items

       blockDimZ (:py:obj:`~.int`) -- *IN*:
           Z grid dimension specified in work-items

       sharedMemBytes (:py:obj:`~.int`) -- *IN*:
           Amount of dynamic shared memory to allocate for this kernel. The
           HIP-Clang compiler provides support for extern shared declarations.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream where the kernel should be dispatched.  May be 0, in which case th
           default stream is used with associated synchronization rules.

       kernelParams (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Kernel parameters to launch

       extra (:py:obj:`~.rocm.bindings._hip_helpers.HipModuleLaunchKernel_extra`/:py:obj:`~.object`) -- *IN*:
           Pointer to kernel arguments.   These are passed directly to the kernel and
           must be in the memory layout and alignment expected by the kernel.
           All passed arguments must be naturally aligned according to their type. The memory address of
           each argument should be a multiple of its size in bytes. Please refer to
           hip_porting_driver_api.md for sample usage.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void ** kernelParams, void ** extra)


.. py:function:: hipModuleLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams)

   launches kernel f with launch parameters and shared memory on stream with arguments passed
   to kernelParams, where thread blocks can cooperate and synchronize as they execute

   Please note, HIP does not support kernel launch with total work items defined in dimension with
   size :math:`gridDim \cdot blockDim \geq 2^{32}`.

   Args:
       f (:py:obj:`~.ihipModuleSymbol_t`/:py:obj:`~.object`) -- *IN*:
           Kernel to launch.

       gridDimX (:py:obj:`~.int`) -- *IN*:
           X grid dimension specified as multiple of blockDimX.

       gridDimY (:py:obj:`~.int`) -- *IN*:
           Y grid dimension specified as multiple of blockDimY.

       gridDimZ (:py:obj:`~.int`) -- *IN*:
           Z grid dimension specified as multiple of blockDimZ.

       blockDimX (:py:obj:`~.int`) -- *IN*:
           X block dimension specified in work-items.

       blockDimY (:py:obj:`~.int`) -- *IN*:
           Y block dimension specified in work-items.

       blockDimZ (:py:obj:`~.int`) -- *IN*:
           Z block dimension specified in work-items.

       sharedMemBytes (:py:obj:`~.int`) -- *IN*:
           Amount of dynamic shared memory to allocate for this kernel. The
           HIP-Clang compiler provides support for extern shared declarations.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream where the kernel should be dispatched. May be 0,
           in which case the default stream is used with associated synchronization rules.

       kernelParams (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           A list of kernel arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorInvalidImage`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorInvalidConfiguration`, :py:obj:`~.hipErrorLaunchFailure`, :py:obj:`~.hipErrorLaunchOutOfResources`,
           :py:obj:`~.hipErrorLaunchTimeOut`, :py:obj:`~.hipErrorCooperativeLaunchTooLarge`, :py:obj:`~.hipErrorSharedObjectInitFailed`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void ** kernelParams)


.. py:function:: hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)

   Launches kernels on multiple devices where thread blocks can cooperate and
   synchronize as they execute.

   Args:
       launchParamsList (:py:obj:`~.hipFunctionLaunchParams_t`/:py:obj:`~.object`) -- *IN*:
           List of launch parameters, one per device.

       numDevices (:py:obj:`~.int`) -- *IN*:
           Size of the launchParamsList array.

       flags (:py:obj:`~.int`) -- *IN*:
           Flags to control launch behavior.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorDeinitialized`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidContext`,
           :py:obj:`~.hipErrorInvalidHandle`, :py:obj:`~.hipErrorInvalidImage`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorInvalidConfiguration`, :py:obj:`~.hipErrorInvalidResourceHandle`, :py:obj:`~.hipErrorLaunchFailure`,
           :py:obj:`~.hipErrorLaunchOutOfResources`, :py:obj:`~.hipErrorLaunchTimeOut`, :py:obj:`~.hipErrorCooperativeLaunchTooLarge`,
           :py:obj:`~.hipErrorSharedObjectInitFailed`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams * launchParamsList, unsigned int numDevices, unsigned int flags)


.. py:function:: hipLaunchCooperativeKernel(f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream)

   Launches kernel f with launch parameters and shared memory on stream with arguments passed
   to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute.

   Please note, HIP does not support kernel launch with total work items defined in dimension with
   size :math:`gridDim \cdot blockDim \geq 2^{32}`.

   Args:
       f (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Kernel to launch.

       gridDim (:py:obj:`~.dim3`) -- *IN*:
           - Grid dimensions specified as multiple of blockDim.

       blockDimX (:py:obj:`~.dim3`) -- *IN*:
           - Block dimensions specified in work-items

       kernelParams (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer of arguments passed to the kernel. If the kernel has multiple
           parameters, 'kernelParams' should be array of pointers, each points the corresponding argument.

       sharedMemBytes (:py:obj:`~.int`) -- *IN*:
           - Amount of dynamic shared memory to allocate for this kernel. The
           HIP-Clang compiler provides support for extern shared declarations.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream where the kernel should be dispatched.  May be 0, in which case th
           default stream is used with associated synchronization rules.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorCooperativeLaunchTooLarge`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLaunchCooperativeKernel(const void * f, dim3 gridDim, dim3 blockDimX, void ** kernelParams, unsigned int sharedMemBytes, hipStream_t stream)


.. py:function:: hipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)

   Launches kernels on multiple devices where thread blocks can cooperate and
   synchronize as they execute.

   Args:
       launchParamsList (:py:obj:`~.hipLaunchParams_t`/:py:obj:`~.object`) -- *IN*:
           List of launch parameters, one per device.

       numDevices (:py:obj:`~.int`) -- *IN*:
           Size of the launchParamsList array.

       flags (:py:obj:`~.int`) -- *IN*:
           Flags to control launch behavior.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorCooperativeLaunchTooLarge`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams * launchParamsList, int numDevices, unsigned int flags)


.. py:function:: hipExtLaunchMultiKernelMultiDevice(launchParamsList, numDevices, flags)

   Launches kernels on multiple devices and guarantees all specified kernels are dispatched
   on respective streams before enqueuing any other work on the specified streams from any other
   threads

   Args:
       launchParamsList (:py:obj:`~.hipLaunchParams_t`/:py:obj:`~.object`) -- *IN*:
           List of launch parameters, one per device.

       numDevices (:py:obj:`~.int`) -- *IN*:
           Size of the launchParamsList array.

       flags (:py:obj:`~.int`) -- *IN*:
           Flags to control launch behavior.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams * launchParamsList, int numDevices, unsigned int flags)


.. py:function:: hipLaunchKernelExC(config, fPtr, args)

   Launches a HIP kernel using a generic function pointer and the specified configuration.

   This function is equivalent to hipLaunchKernelEx but accepts the kernel as a generic function
   pointer.

   Args:
       config (:py:obj:`~.hipLaunchConfig_st`/:py:obj:`~.object`) -- *IN*:
           Pointer to the kernel launch configuration structure.

       fPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the device kernel function.

       args (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of pointers to the kernel arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess` if the kernel is launched successfully, otherwise an appropriate error code.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLaunchKernelExC(const hipLaunchConfig_t * config, const void * fPtr, void ** args)


.. py:function:: hipDrvLaunchKernelEx(config, f, params, extra)

   Launches a HIP kernel using the driver API with the specified configuration.

   This function dispatches the device kernel represented by a HIP function object.
   It passes both the kernel parameters and any extra configuration arguments to the kernel launch.

   Args:
       config (:py:obj:`~.HIP_LAUNCH_CONFIG_st`/:py:obj:`~.object`) -- *IN*:
           Pointer to the kernel launch configuration structure.

       f (:py:obj:`~.ihipModuleSymbol_t`/:py:obj:`~.object`) -- *IN*:
           HIP function object representing the device kernel to be launched.

       params (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of pointers to the kernel parameters.

       extra (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of pointers for additional launch parameters or extra configuration
           data.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess` if the kernel is launched successfully, otherwise an appropriate error code.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG * config, hipFunction_t f, void ** params, void ** extra)


.. py:function:: hipMemGetHandleForAddressRange(handle, dptr, size, handleType, flags)

   Returns a handle for the address range requested.

   This function returns a handle to a device pointer created using either hipMalloc set of APIs
   or through hipMemAddressReserve (as long as the ptr is mapped).

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Ptr to the handle where the fd or other types will be returned.

       dptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device ptr for which we get the handle.

       size (:py:obj:`~.int`) -- *IN*:
           Size of the address range.

       handleType (:py:obj:`~.hipMemRangeHandleType`) -- *IN*:
           Type of the handle requested for the address range.

       flags (:py:obj:`~.int`) -- *IN*:
           Any flags set regarding the handle requested.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess` if the kernel is launched successfully, otherwise an appropriate error code.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemGetHandleForAddressRange(void * handle, hipDeviceptr_t dptr, size_t size, hipMemRangeHandleType handleType, unsigned long long flags)


.. py:function:: hipModuleOccupancyMaxPotentialBlockSize(f, dynSharedMemPerBlk, blockSizeLimit)

   determine the grid and block sizes to achieves maximum occupancy for a kernel

   *  This section describes the occupancy functions of HIP runtime API.

   Please note, HIP does not support kernel launch with total work items defined in dimension with
   size gridDim x blockDim >= 2^32.

   Args:
       f (:py:obj:`~.ihipModuleSymbol_t`/:py:obj:`~.object`) -- *IN*:
           kernel function for which occupancy is calculated

       dynSharedMemPerBlk (:py:obj:`~.int`) -- *IN*:
           dynamic shared memory usage (in bytes) intended for each block

       blockSizeLimit (:py:obj:`~.int`) -- *IN*:
           the maximum block size for the kernel, use 0 for no limit

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               minimum grid size for maximum potential occupancy
       * :py:obj:`~.int`:
               block size for maximum potential occupancy

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleOccupancyMaxPotentialBlockSize(int * gridSize, int * blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit)


.. py:function:: hipModuleOccupancyMaxPotentialBlockSizeWithFlags(f, dynSharedMemPerBlk, blockSizeLimit, flags)

   determine the grid and block sizes to achieves maximum occupancy for a kernel

   Please note, HIP does not support kernel launch with total work items defined in dimension with
   size gridDim x blockDim >= 2^32.

   Args:
       f (:py:obj:`~.ihipModuleSymbol_t`/:py:obj:`~.object`) -- *IN*:
           kernel function for which occupancy is calculated

       dynSharedMemPerBlk (:py:obj:`~.int`) -- *IN*:
           dynamic shared memory usage (in bytes) intended for each block

       blockSizeLimit (:py:obj:`~.int`) -- *IN*:
           the maximum block size for the kernel, use 0 for no limit

       flags (:py:obj:`~.int`) -- *IN*:
           Extra flags for occupancy calculation (only default supported)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               minimum grid size for maximum potential occupancy
       * :py:obj:`~.int`:
               block size for maximum potential occupancy

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int * gridSize, int * blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit, unsigned int flags)


.. py:function:: hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(f, blockSize, dynSharedMemPerBlk)

   Returns occupancy for a device function.

   Args:
       f (:py:obj:`~.ihipModuleSymbol_t`/:py:obj:`~.object`) -- *IN*:
           Kernel function (hipFunction) for which occupancy is calculated

       blockSize (:py:obj:`~.int`) -- *IN*:
           Block size the kernel is intended to be launched with

       dynSharedMemPerBlk (:py:obj:`~.int`) -- *IN*:
           Dynamic shared memory usage (in bytes) intended for each block

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Returned occupancy

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk)


.. py:function:: hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(f, blockSize, dynSharedMemPerBlk, flags)

   Returns occupancy for a device function.

   Args:
       f (:py:obj:`~.ihipModuleSymbol_t`/:py:obj:`~.object`) -- *IN*:
           Kernel function(hipFunction_t) for which occupancy is calculated

       blockSize (:py:obj:`~.int`) -- *IN*:
           Block size the kernel is intended to be launched with

       dynSharedMemPerBlk (:py:obj:`~.int`) -- *IN*:
           Dynamic shared memory usage (in bytes) intended for each block

       flags (:py:obj:`~.int`) -- *IN*:
           Extra flags for occupancy calculation (only default supported)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Returned occupancy

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags)


.. py:function:: hipOccupancyMaxActiveBlocksPerMultiprocessor(f, blockSize, dynSharedMemPerBlk)

   Returns occupancy for a device function.

   Args:
       f (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Kernel function for which occupancy is calculated

       blockSize (:py:obj:`~.int`) -- *IN*:
           Block size the kernel is intended to be launched with

       dynSharedMemPerBlk (:py:obj:`~.int`) -- *IN*:
           Dynamic shared memory usage (in bytes) intended for each block

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDeviceFunction`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Returned occupancy

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * f, int blockSize, size_t dynSharedMemPerBlk)


.. py:function:: hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(f, blockSize, dynSharedMemPerBlk, flags)

   Returns occupancy for a device function.

   Args:
       f (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Kernel function for which occupancy is calculated

       blockSize (:py:obj:`~.int`) -- *IN*:
           Block size the kernel is intended to be launched with

       dynSharedMemPerBlk (:py:obj:`~.int`) -- *IN*:
           Dynamic shared memory usage (in bytes) intended for each block

       flags (:py:obj:`~.int`) -- *IN*:
           Extra flags for occupancy calculation (currently ignored)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDeviceFunction`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Returned occupancy

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags)


.. py:function:: hipOccupancyMaxPotentialBlockSize(f, dynSharedMemPerBlk, blockSizeLimit)

   determine the grid and block sizes to achieves maximum occupancy for a kernel

   Please note, HIP does not support kernel launch with total work items defined in dimension with
   size gridDim x blockDim >= 2^32.

   Args:
       f (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           kernel function for which occupancy is calculated

       dynSharedMemPerBlk (:py:obj:`~.int`) -- *IN*:
           dynamic shared memory usage (in bytes) intended for each block

       blockSizeLimit (:py:obj:`~.int`) -- *IN*:
           the maximum block size for the kernel, use 0 for no limit

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               minimum grid size for maximum potential occupancy
       * :py:obj:`~.int`:
               block size for maximum potential occupancy

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipOccupancyMaxPotentialBlockSize(int * gridSize, int * blockSize, const void * f, size_t dynSharedMemPerBlk, int blockSizeLimit)


.. py:function:: hipOccupancyAvailableDynamicSMemPerBlock(f, numBlocks, blockSize)

   Returns dynamic shared memory available per block when launching numBlocks blocks on SM.

   Returns in ``*dynamicSmemSize`` the maximum size of dynamic shared memory /
   to allow numBlocks blocks per SM.

   Args:
       f (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Kernel function for which occupancy is calculated.

       numBlocks (:py:obj:`~.int`) -- *IN*:
           Number of blocks to fit on SM

       blockSize (:py:obj:`~.int`) -- *IN*:
           Size of the block

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`, :py:obj:`~.hipErrorInvalidDeviceFunction`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorUnknown`
       * :py:obj:`~.int`:
               Returned maximum dynamic shared memory.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * f, int numBlocks, int blockSize)


.. py:function:: hipOccupancyMaxActiveClusters(f, config)

   determines the amount of active kernel clusters can co-exist at the same time in a device

   Args:
       f (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           kernel function for which occupancy is calculated

       config (:py:obj:`~.hipLaunchConfig_st`/:py:obj:`~.object`) -- *IN*:
           pointer to the kernel launch configuration structure

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDeviceFunction`, hipErrorInvalidClusterSize,
           :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               the amount of clusters

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipOccupancyMaxActiveClusters(int * numClusters, const void * f, const hipLaunchConfig_t * config)


.. py:function:: hipOccupancyMaxPotentialClusterSize(f, config)

   returns the maximum cluster size (in number of blocks) that can run on the device

   Args:
       f (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           kernel function for which occupancy is calculated

       config (:py:obj:`~.hipLaunchConfig_st`/:py:obj:`~.object`) -- *IN*:
           pointer to the kernel launch configuration structure

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDeviceFunction`, hipErrorInvalidClusterSize,
           :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               the maximum cluster size

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipOccupancyMaxPotentialClusterSize(int * clusterSize, const void * f, const hipLaunchConfig_t * config)


.. py:function:: hipProfilerStart()

   Start recording of profiling information [Deprecated]
   When using this API, start the profiler with profiling disabled.  (--startdisabled)

   Warning:
       hipProfilerStart API is deprecated, use roctracer/rocTX instead.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipProfilerStart()


.. py:function:: hipProfilerStop()

   Stop recording of profiling information [Deprecated]
   When using this API, start the profiler with profiling disabled.  (--startdisabled)

   Warning:
       hipProfilerStart API is deprecated, use roctracer/rocTX instead.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipProfilerStop()


.. py:function:: hipConfigureCall(gridDim, blockDim, sharedMem, stream)

   Configure a kernel launch.

   *  This section describes the API to support the triple-chevron syntax.

   Please note, HIP does not support kernel launch with total work items defined in dimension with
   size gridDim x blockDim >= 2^32.

   Args:
       gridDim (:py:obj:`~.dim3`) -- *IN*:
           grid dimension specified as multiple of blockDim.

       blockDim (:py:obj:`~.dim3`) -- *IN*:
           block dimensions specified in work-items

       sharedMem (:py:obj:`~.int`) -- *IN*:
           Amount of dynamic shared memory to allocate for this kernel. The
           HIP-Clang compiler provides support for extern shared declarations.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream where the kernel should be dispatched.  May be 0, in which case the
           default stream is used with associated synchronization rules.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream)


.. py:function:: hipSetupArgument(arg, size, offset)

   Set a kernel argument.

   Args:
       arg (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer the argument in host memory.

       size (:py:obj:`~.int`) -- *IN*:
           Size of the argument.

       offset (:py:obj:`~.int`) -- *IN*:
           Offset of the argument on the argument stack.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipSetupArgument(const void * arg, size_t size, size_t offset)


.. py:function:: hipLaunchByPtr(func)

   Launch a kernel.

   Args:
       func (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Kernel to launch.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLaunchByPtr(const void * func)


.. py:function:: hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream)

   C compliant kernel launch API

   Args:
       function_address (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Kernel stub function pointer.

       numBlocks (:py:obj:`~.dim3`) -- *IN*:
           - Number of blocks.

       dimBlocks (:py:obj:`~.dim3`) -- *IN*:
           - Dimension of a block

       args (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer of arguments passed to the kernel. If the kernel has multiple
           parameters, 'args' should be array of pointers, each points the corresponding argument.

       sharedMemBytes (:py:obj:`~.int`) -- *IN*:
           - Amount of dynamic shared memory to allocate for this kernel. The
           HIP-Clang compiler provides support for extern shared declarations.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream where the kernel should be dispatched.  May be 0, in which case th
           default stream is used with associated synchronization rules.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLaunchKernel(const void * function_address, dim3 numBlocks, dim3 dimBlocks, void ** args, size_t sharedMemBytes, hipStream_t stream)


.. py:function:: hipLaunchHostFunc(stream, fn, userData)

   Enqueues a host function call in a stream.

   The host function to call in this API will be executed after the preceding operations in
   the stream are complete. The function is a blocking operation that blocks operations in the
   stream that follow it, until the function is returned.
   Event synchronization and internal callback functions make sure enqueued operations will
   execute in order, in the stream.

   The host function must not make any HIP API calls. The host function is non-reentrant. It must
   not perform sychronization with any operation that may depend on other processing execution
   but is not enqueued to run earlier in the stream.

   Host functions that are enqueued respectively in different non-blocking streams can run
   concurrently.

   Warning:
       This API is marked as beta, meaning, while this is feature complete,
       it is still open to changes and may have outstanding issues.

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - The stream to enqueue work in.

       fn (:py:obj:`~.hipHostFn_t`/:py:obj:`~.object`) -- *IN*:
           - The function to call once enqueued preceeding operations are complete.

       userData (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - User-specified data to be passed to the function.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidResourceHandle`, :py:obj:`~.hipErrorInvalidValue`,
           :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void * userData)


.. py:function:: hipDrvMemcpy2DUnaligned(pCopy)

   Copies memory for 2D arrays.

   Args:
       pCopy (:py:obj:`~.hip_Memcpy2D`/:py:obj:`~.object`):
           - Parameters for the memory copy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D * pCopy)


.. py:function:: hipExtLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags)

   Launches kernel from the pointer address, with arguments and shared memory on stream.

   Args:
       function_address (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the Kernel to launch.

       numBlocks (:py:obj:`~.dim3`) -- *IN*:
           -  Number of blocks.

       dimBlocks (:py:obj:`~.dim3`) -- *IN*:
           - Dimension of a block.

       args (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer of arguments passed to the kernel. If the kernel has multiple
           parameters, 'args' should be array of pointers, each points the corresponding argument.

       sharedMemBytes (:py:obj:`~.int`) -- *IN*:
           - Amount of dynamic shared memory to allocate for this kernel.
           HIP-Clang compiler provides support for extern shared declarations.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream where the kernel should be dispatched.
           May be 0, in which case the default stream is used with associated synchronization rules.

       startEvent (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           - If non-null, specified event will be updated to track the start time of
           the kernel launch. The event must be created before calling this API.

       stopEvent (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           - If non-null, specified event will be updated to track the stop time of
           the kernel launch. The event must be created before calling this API.

       flags (:py:obj:`~.int`) -- *IN*:
           - The value of hipExtAnyOrderLaunch, signifies if kernel can be
           launched in any order.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotInitialized`, :py:obj:`~.hipErrorInvalidValue`.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtLaunchKernel(const void * function_address, dim3 numBlocks, dim3 dimBlocks, void ** args, size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent, hipEvent_t stopEvent, int flags)


.. py:function:: hipCreateTextureObject(pResDesc, pTexDesc, pResViewDesc)

   Creates a texture object.

   Note:
       3D linear filter isn't supported on GFX90A boards, on which the API `hipCreateTextureObject` will return hipErrorNotSupported.

   Args:
       pResDesc (:py:obj:`~.hipResourceDesc`/:py:obj:`~.object`) -- *IN*:
           pointer to resource descriptor

       pTexDesc (:py:obj:`~.hipTextureDesc`/:py:obj:`~.object`) -- *IN*:
           pointer to texture descriptor

       pResViewDesc (:py:obj:`~.hipResourceViewDesc`/:py:obj:`~.object`) -- *IN*:
           pointer to resource view descriptor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.__hip_texture`:
               pointer to the texture object to create

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCreateTextureObject(hipTextureObject_t * pTexObject, const hipResourceDesc * pResDesc, const hipTextureDesc * pTexDesc, const struct hipResourceViewDesc * pResViewDesc)


.. py:function:: hipDestroyTextureObject(textureObject)

   Destroys a texture object.

   Args:
       textureObject (:py:obj:`~.__hip_texture`/:py:obj:`~.object`) -- *IN*:
           texture object to destroy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject)


.. py:function:: hipGetChannelDesc(desc)

   Gets the channel descriptor in an array.

   Args:
       desc (:py:obj:`~.hipChannelFormatDesc`/:py:obj:`~.object`) -- *IN*:
           pointer to channel format descriptor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipArray`:
               memory array on the device

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetChannelDesc(hipChannelFormatDesc * desc, hipArray_const_t array)


.. py:function:: hipGetTextureObjectResourceDesc(textureObject)

   Gets resource descriptor for the texture object.

   Args:
       textureObject (:py:obj:`~.__hip_texture`/:py:obj:`~.object`) -- *IN*:
           texture object

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipResourceDesc`:
               pointer to resource descriptor

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc * pResDesc, hipTextureObject_t textureObject)


.. py:function:: hipGetTextureObjectResourceViewDesc(textureObject)

   Gets resource view descriptor for the texture object.

   Args:
       textureObject (:py:obj:`~.__hip_texture`/:py:obj:`~.object`) -- *IN*:
           texture object

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipResourceViewDesc`:
               pointer to resource view descriptor

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc * pResViewDesc, hipTextureObject_t textureObject)


.. py:function:: hipGetTextureObjectTextureDesc(textureObject)

   Gets texture descriptor for the texture object.

   Args:
       textureObject (:py:obj:`~.__hip_texture`/:py:obj:`~.object`) -- *IN*:
           texture object

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipTextureDesc`:
               pointer to texture descriptor

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc * pTexDesc, hipTextureObject_t textureObject)


.. py:function:: hipTexObjectCreate(pResDesc, pTexDesc, pResViewDesc)

   Creates a texture object.

   Args:
       pResDesc (:py:obj:`~.HIP_RESOURCE_DESC_st`/:py:obj:`~.object`) -- *IN*:
           pointer to resource descriptor

       pTexDesc (:py:obj:`~.HIP_TEXTURE_DESC_st`/:py:obj:`~.object`) -- *IN*:
           pointer to texture descriptor

       pResViewDesc (:py:obj:`~.HIP_RESOURCE_VIEW_DESC_st`/:py:obj:`~.object`) -- *IN*:
           pointer to resource view descriptor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.__hip_texture`:
               pointer to texture object to create

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexObjectCreate(hipTextureObject_t * pTexObject, const HIP_RESOURCE_DESC * pResDesc, const HIP_TEXTURE_DESC * pTexDesc, const HIP_RESOURCE_VIEW_DESC * pResViewDesc)


.. py:function:: hipTexObjectDestroy(texObject)

   Destroys a texture object.

   Args:
       texObject (:py:obj:`~.__hip_texture`/:py:obj:`~.object`) -- *IN*:
           texture object to destroy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexObjectDestroy(hipTextureObject_t texObject)


.. py:function:: hipTexObjectGetResourceDesc(texObject)

   Gets resource descriptor of a texture object.

   Args:
       texObject (:py:obj:`~.__hip_texture`/:py:obj:`~.object`) -- *IN*:
           texture object

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotSupported`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.HIP_RESOURCE_DESC_st`:
               pointer to resource descriptor

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC * pResDesc, hipTextureObject_t texObject)


.. py:function:: hipTexObjectGetResourceViewDesc(texObject)

   Gets resource view descriptor of a texture object.

   Args:
       texObject (:py:obj:`~.__hip_texture`/:py:obj:`~.object`) -- *IN*:
           texture object

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotSupported`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.HIP_RESOURCE_VIEW_DESC_st`:
               pointer to resource view descriptor

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC * pResViewDesc, hipTextureObject_t texObject)


.. py:function:: hipTexObjectGetTextureDesc(texObject)

   Gets texture descriptor of a texture object.

   Args:
       texObject (:py:obj:`~.__hip_texture`/:py:obj:`~.object`) -- *IN*:
           texture object

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotSupported`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.HIP_TEXTURE_DESC_st`:
               pointer to texture descriptor

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC * pTexDesc, hipTextureObject_t texObject)


.. py:function:: hipMallocMipmappedArray(desc, extent, numLevels, flags)

   Allocate a mipmapped array on the device.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       desc (:py:obj:`~.hipChannelFormatDesc`/:py:obj:`~.object`) -- *IN*:
           - Requested channel format

       extent (:py:obj:`~.hipExtent`) -- *IN*:
           - Requested allocation size (width field in elements)

       numLevels (:py:obj:`~.int`) -- *IN*:
           - Number of mipmap levels to allocate

       flags (:py:obj:`~.int`) -- *IN*:
           - Flags for extensions

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorMemoryAllocation`
       * :py:obj:`~.hipMipmappedArray`:
               - Pointer to allocated mipmapped array in device memory

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMallocMipmappedArray(hipMipmappedArray_t * mipmappedArray, const struct hipChannelFormatDesc * desc, struct hipExtent extent, unsigned int numLevels, unsigned int flags)


.. py:function:: hipFreeMipmappedArray(mipmappedArray)

   Frees a mipmapped array on the device.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       mipmappedArray (:py:obj:`~.hipMipmappedArray`/:py:obj:`~.object`) -- *IN*:
           - Pointer to mipmapped array to free

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray)


.. py:function:: hipGetMipmappedArrayLevel(mipmappedArray, level)

   Gets a mipmap level of a HIP mipmapped array.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       mipmappedArray (:py:obj:`~.hipMipmappedArray`/:py:obj:`~.object`) -- *IN*:
           - HIP mipmapped array

       level (:py:obj:`~.int`) -- *IN*:
           - Mipmap level

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipArray`:
               - Returned mipmap level HIP array

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetMipmappedArrayLevel(hipArray_t * levelArray, hipMipmappedArray_const_t mipmappedArray, unsigned int level)


.. py:function:: hipMipmappedArrayCreate(pMipmappedArrayDesc, numMipmapLevels)

   Create a mipmapped array.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       pMipmappedArrayDesc (:py:obj:`~.HIP_ARRAY3D_DESCRIPTOR`/:py:obj:`~.object`) -- *IN*:
           mipmapped array descriptor

       numMipmapLevels (:py:obj:`~.int`) -- *IN*:
           mipmap level

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorNotSupported`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipMipmappedArray`:
               pointer to mipmapped array

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t * pHandle, HIP_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int numMipmapLevels)


.. py:function:: hipMipmappedArrayDestroy()

   Destroy a mipmapped array.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipMipmappedArray`:
               pointer to mipmapped array to destroy

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray)


.. py:function:: hipMipmappedArrayGetLevel(pLevelArray, level)

   Get a mipmapped array on a mipmapped level.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       pLevelArray (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer of array

       level (:py:obj:`~.int`) -- *OUT*:
           Mipmap level

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipMipmappedArray`:
               Pointer of mipmapped array on the requested mipmap level

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMipmappedArrayGetLevel(hipArray_t * pLevelArray, hipMipmappedArray_t hMipMappedArray, unsigned int level)


.. py:function:: hipBindTextureToMipmappedArray(tex, mipmappedArray, desc)

   Binds a mipmapped array to a texture [Deprecated]

   Args:
       tex (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           pointer to the texture reference to bind

       mipmappedArray (:py:obj:`~.hipMipmappedArray`/:py:obj:`~.object`) -- *IN*:
           memory mipmapped array on the device

       desc (:py:obj:`~.hipChannelFormatDesc`/:py:obj:`~.object`) -- *IN*:
           opointer to the channel format

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipBindTextureToMipmappedArray(const textureReference * tex, hipMipmappedArray_const_t mipmappedArray, const hipChannelFormatDesc * desc)


.. py:function:: hipGetTextureReference(symbol)

   Gets the texture reference related with the symbol [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the symbol related with the texture for the reference

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.textureReference`:
               texture reference

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetTextureReference(const textureReference ** texref, const void * symbol)


.. py:function:: hipTexRefGetBorderColor(texRef)

   Gets the border color used by a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.float`:
               Returned Type and Value of RGBA color.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetBorderColor(float * pBorderColor, const textureReference * texRef)


.. py:function:: hipTexRefGetArray(pArray, texRef)

   Gets the array bound to a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       pArray (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Returned array.

       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetArray(hipArray_t * pArray, const textureReference * texRef)


.. py:function:: hipTexRefSetAddressMode(texRef, dim, am)

   Sets address mode for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           texture reference.

       dim (:py:obj:`~.int`) -- *IN*:
           Dimension of the texture.

       am (:py:obj:`~.hipTextureAddressMode`) -- *IN*:
           Value of the texture address mode.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetAddressMode(textureReference * texRef, int dim, enum hipTextureAddressMode am)


.. py:function:: hipTexRefSetArray(tex, array, flags)

   Binds an array as a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       tex (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer texture reference.

       array (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Array to bind.

       flags (:py:obj:`~.int`) -- *IN*:
           Flags should be set as HIP_TRSA_OVERRIDE_FORMAT, as a valid value.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetArray(textureReference * tex, hipArray_const_t array, unsigned int flags)


.. py:function:: hipTexRefSetFilterMode(texRef, fm)

   Set filter mode for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer texture reference.

       fm (:py:obj:`~.hipTextureFilterMode`) -- *IN*:
           Value of texture filter mode.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetFilterMode(textureReference * texRef, enum hipTextureFilterMode fm)


.. py:function:: hipTexRefSetFlags(texRef, Flags)

   Set flags for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer texture reference.

       Flags (:py:obj:`~.int`) -- *IN*:
           Value of flags.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetFlags(textureReference * texRef, unsigned int Flags)


.. py:function:: hipTexRefSetFormat(texRef, fmt, NumPackedComponents)

   Set format for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer texture reference.

       fmt (:py:obj:`~.hipArray_Format`) -- *IN*:
           Value of format.

       NumPackedComponents (:py:obj:`~.int`) -- *IN*:
           Number of components per array.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetFormat(textureReference * texRef, hipArray_Format fmt, int NumPackedComponents)


.. py:function:: hipBindTexture(tex, devPtr, desc, size)

   Binds a memory area to a texture [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       tex (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Texture to bind.

       devPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer of memory on the device.

       desc (:py:obj:`~.hipChannelFormatDesc`/:py:obj:`~.object`) -- *IN*:
           Pointer of channel format descriptor.

       size (:py:obj:`~.int`) -- *IN*:
           Size of memory in bites.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               Offset in bytes.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipBindTexture(size_t * offset, const textureReference * tex, const void * devPtr, const hipChannelFormatDesc * desc, size_t size)


.. py:function:: hipBindTexture2D(tex, devPtr, desc, width, height, pitch)

   Binds a 2D memory area to a texture [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       tex (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Texture to bind.

       devPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer of 2D memory area on the device.

       desc (:py:obj:`~.hipChannelFormatDesc`/:py:obj:`~.object`) -- *IN*:
           Pointer of channel format descriptor.

       width (:py:obj:`~.int`) -- *IN*:
           Width in texel units.

       height (:py:obj:`~.int`) -- *IN*:
           Height in texel units.

       pitch (:py:obj:`~.int`) -- *IN*:
           Pitch in bytes.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               Offset in bytes.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipBindTexture2D(size_t * offset, const textureReference * tex, const void * devPtr, const hipChannelFormatDesc * desc, size_t width, size_t height, size_t pitch)


.. py:function:: hipBindTextureToArray(tex, array, desc)

   Binds a memory area to a texture [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       tex (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

       array (:py:obj:`~.hipArray`/:py:obj:`~.object`) -- *IN*:
           Array to bind.

       desc (:py:obj:`~.hipChannelFormatDesc`/:py:obj:`~.object`) -- *IN*:
           Pointer of channel format descriptor.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipBindTextureToArray(const textureReference * tex, hipArray_const_t array, const hipChannelFormatDesc * desc)


.. py:function:: hipGetTextureAlignmentOffset(texref)

   Get the offset of the alignment in a texture [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texref (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               Offset in bytes.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetTextureAlignmentOffset(size_t * offset, const textureReference * texref)


.. py:function:: hipUnbindTexture(tex)

   Unbinds a texture [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       tex (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Texture to unbind.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipUnbindTexture(const textureReference * tex)


.. py:function:: hipTexRefGetAddress(texRef)

   Gets the address for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer of device address.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetAddress(hipDeviceptr_t * dev_ptr, const textureReference * texRef)


.. py:function:: hipTexRefGetAddressMode(texRef, dim)

   Gets the address mode for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

       dim (:py:obj:`~.int`) -- *IN*:
           Dimension.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipTextureAddressMode`:
               Pointer of address mode.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode * pam, const textureReference * texRef, int dim)


.. py:function:: hipTexRefGetFilterMode(texRef)

   Gets filter mode for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipTextureFilterMode`:
               Pointer of filter mode.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode * pfm, const textureReference * texRef)


.. py:function:: hipTexRefGetFlags(texRef)

   Gets flags for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               Pointer of flags.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetFlags(unsigned int * pFlags, const textureReference * texRef)


.. py:function:: hipTexRefGetFormat(texRef)

   Gets texture format for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipArray_Format`:
               Pointer of the format.
       * :py:obj:`~.int`:
               Pointer of number of channels.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetFormat(hipArray_Format * pFormat, int * pNumChannels, const textureReference * texRef)


.. py:function:: hipTexRefGetMaxAnisotropy(texRef)

   Gets the maximum anisotropy for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               Pointer of the maximum anisotropy.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetMaxAnisotropy(int * pmaxAnsio, const textureReference * texRef)


.. py:function:: hipTexRefGetMipmapFilterMode(texRef)

   Gets the mipmap filter mode for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipTextureFilterMode`:
               Pointer of the mipmap filter mode.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode * pfm, const textureReference * texRef)


.. py:function:: hipTexRefGetMipmapLevelBias(texRef)

   Gets the mipmap level bias for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.float`:
               Pointer of the mipmap level bias.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetMipmapLevelBias(float * pbias, const textureReference * texRef)


.. py:function:: hipTexRefGetMipmapLevelClamp(texRef)

   Gets the minimum and maximum mipmap level clamps for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.float`:
               Pointer of the minimum mipmap level clamp.
       * :py:obj:`~.float`:
               Pointer of the maximum mipmap level clamp.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, const textureReference * texRef)


.. py:function:: hipTexRefGetMipMappedArray(texRef)

   Gets the mipmapped array bound to a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipMipmappedArray`:
               Pointer of the mipmapped array.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t * pArray, const textureReference * texRef)


.. py:function:: hipTexRefSetAddress(texRef, dptr, bytes)

   Sets an bound address for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

       dptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer of device address to bind.

       bytes (:py:obj:`~.int`) -- *IN*:
           Size in bytes.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               Pointer of the offset in bytes.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetAddress(size_t * ByteOffset, textureReference * texRef, hipDeviceptr_t dptr, size_t bytes)


.. py:function:: hipTexRefSetAddress2D(texRef, desc, dptr, Pitch)

   Set a bind an address as a 2D texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

       desc (:py:obj:`~.HIP_ARRAY_DESCRIPTOR`/:py:obj:`~.object`) -- *IN*:
           Pointer of array descriptor.

       dptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer of device address to bind.

       Pitch (:py:obj:`~.int`) -- *IN*:
           Pitch in bytes.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetAddress2D(textureReference * texRef, const HIP_ARRAY_DESCRIPTOR * desc, hipDeviceptr_t dptr, size_t Pitch)


.. py:function:: hipTexRefSetMaxAnisotropy(texRef, maxAniso)

   Sets the maximum anisotropy for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

       maxAniso (:py:obj:`~.int`) -- *OUT*:
           Value of the maximum anisotropy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetMaxAnisotropy(textureReference * texRef, unsigned int maxAniso)


.. py:function:: hipTexRefSetBorderColor(texRef)

   Sets border color for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.float`:
               Pointer of border color.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetBorderColor(textureReference * texRef, float * pBorderColor)


.. py:function:: hipTexRefSetMipmapFilterMode(texRef, fm)

   Sets mipmap filter mode for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

       fm (:py:obj:`~.hipTextureFilterMode`) -- *IN*:
           Value of filter mode.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetMipmapFilterMode(textureReference * texRef, enum hipTextureFilterMode fm)


.. py:function:: hipTexRefSetMipmapLevelBias(texRef, bias)

   Sets mipmap level bias for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

       bias (:py:obj:`~.float`/:py:obj:`~.int`) -- *IN*:
           Value of mipmap bias.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetMipmapLevelBias(textureReference * texRef, float bias)


.. py:function:: hipTexRefSetMipmapLevelClamp(texRef, minMipMapLevelClamp, maxMipMapLevelClamp)

   Sets mipmap level clamp for a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference.

       minMipMapLevelClamp (:py:obj:`~.float`/:py:obj:`~.int`) -- *IN*:
           Value of minimum mipmap level clamp.

       maxMipMapLevelClamp (:py:obj:`~.float`/:py:obj:`~.int`) -- *IN*:
           Value of maximum mipmap level clamp.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetMipmapLevelClamp(textureReference * texRef, float minMipMapLevelClamp, float maxMipMapLevelClamp)


.. py:function:: hipTexRefSetMipmappedArray(texRef, mipmappedArray, Flags)

   Binds mipmapped array to a texture reference [Deprecated]

   Warning:
       This API is deprecated.

   Args:
       texRef (:py:obj:`~.textureReference`/:py:obj:`~.object`) -- *IN*:
           Pointer of texture reference to bind.

       mipmappedArray (:py:obj:`~.hipMipmappedArray`/:py:obj:`~.object`) -- *IN*:
           Pointer of mipmapped array to bind.

       Flags (:py:obj:`~.int`) -- *IN*:
           Flags should be set as HIP_TRSA_OVERRIDE_FORMAT, as a valid value.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipTexRefSetMipmappedArray(textureReference * texRef, struct hipMipmappedArray * mipmappedArray, unsigned int Flags)


.. py:function:: hipApiName(id)

   Returns HIP API name by ID.

   *  This section describes the callback/Activity of HIP runtime API.

   Args:
       id (:py:obj:`~.int`) -- *IN*:
           ID of HIP API

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`:
               Always returns `~.hipError_t.hipSuccess`.
       * :py:obj:`~.bytes`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       const char * hipApiName(uint32_t id)


.. py:function:: hipKernelNameRef(f)

   Returns kernel name reference by function name.

   Args:
       f (:py:obj:`~.ihipModuleSymbol_t`/:py:obj:`~.object`) -- *IN*:
           Name of function

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`:
               Always returns `~.hipError_t.hipSuccess`.
       * :py:obj:`~.bytes`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       const char * hipKernelNameRef(const hipFunction_t f)


.. py:function:: hipKernelNameRefByPtr(hostFunction, stream)

   Retrives kernel for a given host pointer, unless stated otherwise.

   Args:
       hostFunction (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer of host function.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream the kernel is executed on.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`:
               Always returns `~.hipError_t.hipSuccess`.
       * :py:obj:`~.bytes`: The name of the passed kernel function object, or nullptr.

   .. rubric:: C signature

   .. code-block:: c

       const char * hipKernelNameRefByPtr(const void * hostFunction, hipStream_t stream)


.. py:function:: hipGetStreamDeviceId(stream)

   Returns device ID on the stream.

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream of device executed on.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`:
               Always returns `~.hipError_t.hipSuccess`.
       * :py:obj:`~.int`: The device ID on the stream.

   .. rubric:: C signature

   .. code-block:: c

       int hipGetStreamDeviceId(hipStream_t stream)


.. py:function:: hipStreamBeginCapture(stream, mode)

   Begins graph capture on a stream.

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream to initiate capture.

       mode (:py:obj:`~.hipStreamCaptureMode`) -- *IN*:
           - Controls the interaction of this capture sequence with other API calls that
           are not safe.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode)


.. py:function:: hipStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData, numDependencies, mode)

   Begins graph capture on a stream to an existing graph.

   Warning:
       param "const hipGraphEdgeData* dependencyData" is currently not supported and has to be
       passed as nullptr. This API is marked as beta, meaning, while this is feature complete, it is still
       open to changes and may have outstanding issues.

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream to initiate capture.

       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Graph to capture into.

       dependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Dependencies of the first node captured in the stream. Can be NULL if
           numDependencies is 0.

       dependencyData (:py:obj:`~.hipGraphEdgeData`/:py:obj:`~.object`) -- *IN*:
           - Optional array of data associated with each dependency.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies.

       mode (:py:obj:`~.hipStreamCaptureMode`) -- *IN*:
           - Controls the interaction of this capture sequence with other API calls that
           are not safe.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph, const hipGraphNode_t * dependencies, const hipGraphEdgeData * dependencyData, size_t numDependencies, hipStreamCaptureMode mode)


.. py:function:: hipStreamEndCapture(stream)

   Ends capture on a stream, returning the captured graph.

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream to end capture.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipGraph`:
               - Captured graph.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t * pGraph)


.. py:function:: hipStreamGetCaptureInfo(stream)

   Get capture status of a stream.

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream of which to get capture status from.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorStreamCaptureImplicit`
       * :py:obj:`~.hipStreamCaptureStatus`:
               - Returns current capture status.
       * :py:obj:`~.int`:
               - Unique capture ID.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus * pCaptureStatus, unsigned long long * pId)


.. py:function:: hipStreamGetCaptureInfo_v2(stream, dependencies_out)

   Get stream's capture state

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream of which to get capture status from.

       dependencies_out (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           - Pointer to an array of nodes representing the graphs
           dependencies.

   Returns:
       A :py:obj:`~.tuple` of size 5 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorStreamCaptureImplicit`
       * :py:obj:`~.hipStreamCaptureStatus`:
               - Returns current capture status.
       * :py:obj:`~.int`:
               - Unique capture ID.
       * :py:obj:`~.ihipGraph`:
               - Returns the graph being captured into.
       * :py:obj:`~.int`:
               - Returns size of the array returned in dependencies_out.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, hipGraph_t * graph_out, const hipGraphNode_t ** dependencies_out, size_t * numDependencies_out)


.. py:function:: hipStreamIsCapturing(stream)

   Get stream's capture state

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream of which to get capture status from.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorStreamCaptureImplicit`
       * :py:obj:`~.hipStreamCaptureStatus`:
               - Returns current capture status.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus * pCaptureStatus)


.. py:function:: hipStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags)

   Update the set of dependencies in a capturing stream

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           Stream that is being captured.

       dependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to an array of nodes to add/replace.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           Size of the dependencies array.

       flags (:py:obj:`~.int`) -- *IN*:
           Flag to update dependency set. Should be one of the values
           in enum :py:obj:`~.hipStreamUpdateCaptureDependenciesFlags`.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorIllegalState`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t * dependencies, size_t numDependencies, unsigned int flags)


.. py:function:: hipThreadExchangeStreamCaptureMode()

   Swaps the stream capture mode of a thread.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipStreamCaptureMode`:
               - Pointer to mode value to swap with the current mode.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode * mode)


.. py:function:: hipGraphCreate(flags)

   Creates a graph

   Args:
       flags (:py:obj:`~.int`) -- *IN*:
           - flags for graph creation, must be 0.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorMemoryAllocation`
       * :py:obj:`~.ihipGraph`:
               - pointer to graph to create.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphCreate(hipGraph_t * pGraph, unsigned int flags)


.. py:function:: hipGraphDestroy(graph)

   Destroys a graph

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - instance of graph to destroy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphDestroy(hipGraph_t graph)


.. py:function:: hipGraphAddDependencies(graph, from_, to, numDependencies)

   Adds dependency edges to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph to add dependencies to.

       from (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the graph nodes with dependencies to add from.

       to (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the graph nodes to add dependencies to.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies to add.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t * from, const hipGraphNode_t * to, size_t numDependencies)


.. py:function:: hipGraphRemoveDependencies(graph, from_, to, numDependencies)

   Removes dependency edges from a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph to remove dependencies from.

       from (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Array of nodes that provide the dependencies.

       to (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Array of dependent nodes.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies to remove.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t * from, const hipGraphNode_t * to, size_t numDependencies)


.. py:function:: hipGraphGetEdges(graph)

   Returns a graph's dependency edges.

   from and to may both be NULL, in which case this function only returns the number of edges in
   numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher than the actual
   number of edges, the remaining entries in from and to will be set to NULL, and the number of
   edges actually returned will be written to numEdges.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph to get the edges from.

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to the graph nodes to return edge endpoints.
       * :py:obj:`~.hipGraphNode`:
               - Pointer to the graph nodes to return edge endpoints.
       * :py:obj:`~.int`:
               - Returns number of edges.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t * from, hipGraphNode_t * to, size_t * numEdges)


.. py:function:: hipGraphGetNodes(graph)

   Returns a graph's nodes.

   nodes may be NULL, in which case this function will return the number of nodes in numNodes.
   Otherwise, numNodes entries will be filled in. If numNodes is higher than the actual number of
   nodes, the remaining entries in nodes will be set to NULL, and the number of nodes actually
   obtained will be returned in numNodes.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to get the nodes from.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to return the  graph nodes.
       * :py:obj:`~.int`:
               - Returns the number of graph nodes.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t * nodes, size_t * numNodes)


.. py:function:: hipGraphGetRootNodes(graph)

   Returns a graph's root nodes.

   pRootNodes may be NULL, in which case this function will return the number of root nodes in
   pNumRootNodes. Otherwise, pNumRootNodes entries will be filled in. If pNumRootNodes is higher
   than the actual number of root nodes, the remaining entries in pRootNodes will be set to NULL,
   and the number of nodes actually obtained will be returned in pNumRootNodes.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph to get the nodes from.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to return the graph's root nodes.
       * :py:obj:`~.int`:
               - Returns the number of graph's root nodes.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t * pRootNodes, size_t * pNumRootNodes)


.. py:function:: hipGraphNodeGetDependencies(node)

   Returns a node's dependencies.

   pDependencies may be NULL, in which case this function will return the number of dependencies in
   pNumDependencies. Otherwise, pNumDependencies entries will be filled in. If pNumDependencies is
   higher than the actual number of dependencies, the remaining entries in pDependencies will be set
   to NULL, and the number of nodes actually obtained will be returned in pNumDependencies.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Graph node to get the dependencies from.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to return the dependencies.
       * :py:obj:`~.int`:
               -  Returns the number of graph node dependencies.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t * pDependencies, size_t * pNumDependencies)


.. py:function:: hipGraphNodeGetDependentNodes(node)

   Returns a node's dependent nodes.

   pDependentNodes may be NULL, in which case this function will return the number of dependent
   nodes in pNumDependentNodes. Otherwise, pNumDependentNodes entries will be filled in. If
   pNumDependentNodes is higher than the actual number of dependent nodes, the remaining entries in
   pDependentNodes will be set to NULL, and the number of nodes actually obtained will be returned
   in pNumDependentNodes.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Graph node to get the dependent nodes from.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to return the graph dependent nodes.
       * :py:obj:`~.int`:
               - Returns the number of graph node dependent nodes.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t * pDependentNodes, size_t * pNumDependentNodes)


.. py:function:: hipGraphNodeGetType(node)

   Returns a node's type.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node to get type of.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNodeType`:
               - Returns the node's type.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType * pType)


.. py:function:: hipGraphDestroyNode(node)

   Remove a node from the graph.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - graph node to remove

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphDestroyNode(hipGraphNode_t node)


.. py:function:: hipGraphClone(originalGraph)

   Clones a graph.

   Args:
       originalGraph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - original graph to clone from.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorMemoryAllocation`
       * :py:obj:`~.ihipGraph`:
               - Returns newly created cloned graph.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphClone(hipGraph_t * pGraphClone, hipGraph_t originalGraph)


.. py:function:: hipGraphNodeFindInClone(originalNode, clonedGraph)

   Finds a cloned version of a node.

   Args:
       originalNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - original node handle.

       clonedGraph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Cloned graph to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Returns the cloned node.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphNodeFindInClone(hipGraphNode_t * pNode, hipGraphNode_t originalNode, hipGraph_t clonedGraph)


.. py:function:: hipGraphInstantiate(graph, bufferSize)

   Creates an executable graph from a graph

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to instantiate.

       bufferSize (:py:obj:`~.int`) -- *OUT*:
           - Size of the log buffer.

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorOutOfMemory`
       * :py:obj:`~.hipGraphExec`:
               - Pointer to instantiated executable graph.
       * :py:obj:`~.hipGraphNode`:
               - Pointer to error node. In case an error occured during
               graph instantiation, it could modify the corresponding node.
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               - Pointer to log buffer.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphInstantiate(hipGraphExec_t * pGraphExec, hipGraph_t graph, hipGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize)


.. py:function:: hipGraphInstantiateWithFlags(graph, flags)

   Creates an executable graph from a graph.

   Warning:
       This API does not support any of flag and is behaving as hipGraphInstantiate.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to instantiate.

       flags (:py:obj:`~.int`) -- *IN*:
           - Flags to control instantiation.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphExec`:
               - Pointer to instantiated executable graph.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t * pGraphExec, hipGraph_t graph, unsigned long long flags)


.. py:function:: hipGraphInstantiateWithParams(graph, instantiateParams)

   Creates an executable graph from a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to instantiate.

       instantiateParams (:py:obj:`~.hipGraphInstantiateParams`/:py:obj:`~.object`) -- *IN*:
           - Graph instantiation Params

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphExec`:
               - Pointer to instantiated executable graph.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphInstantiateWithParams(hipGraphExec_t * pGraphExec, hipGraph_t graph, hipGraphInstantiateParams * instantiateParams)


.. py:function:: hipGraphLaunch(graphExec, stream)

   Launches an executable graph in the specified stream.

   Args:
       graphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of executable graph to launch.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Instance of stream in which to launch executable graph.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream)


.. py:function:: hipGraphUpload(graphExec, stream)

   Uploads an executable graph to a stream

   Args:
       graphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of executable graph to be uploaded.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Instance of stream to which the executable graph is uploaded to.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream)


.. py:function:: hipGraphAddNode(graph, pDependencies, numDependencies, nodeParams)

   Creates a kernel execution node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to add the created node to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the dependencies on the kernel execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies.

       nodeParams (:py:obj:`~.hipGraphNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the node parameters.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`.
       * :py:obj:`~.hipGraphNode`:
               - Pointer to kernel graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipGraphNodeParams * nodeParams)


.. py:function:: hipGraphExecGetFlags(graphExec)

   Return the flags of an executable graph.

   Args:
       graphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Executable graph to get the flags from.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`.
       * :py:obj:`~.int`:
               - Flags used to instantiate this executable graph.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecGetFlags(hipGraphExec_t graphExec, unsigned long long * flags)


.. py:function:: hipGraphNodeSetParams(node, nodeParams)

   Updates parameters of a graph's node.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters for.

       nodeParams (:py:obj:`~.hipGraphNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the parameters to be set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDeviceFunction`,
           :py:obj:`~.hipErrorNotSupported`.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphNodeSetParams(hipGraphNode_t node, hipGraphNodeParams * nodeParams)


.. py:function:: hipGraphExecNodeSetParams(graphExec, node, nodeParams)

   Updates parameters of an executable graph's node.

   Args:
       graphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of the executable graph.

       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters to.

       nodeParams (:py:obj:`~.hipGraphNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the parameters to be set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDeviceFunction`,
           :py:obj:`~.hipErrorNotSupported`.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecNodeSetParams(hipGraphExec_t graphExec, hipGraphNode_t node, hipGraphNodeParams * nodeParams)


.. py:function:: hipGraphExecDestroy(graphExec)

   Destroys an executable graph

   Args:
       graphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of executable graph to destroy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec)


.. py:function:: hipGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out)

   Check whether an executable graph can be updated with a graph and perform the update if  *
   possible.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - instance of executable graph to update.

       hGraph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - graph that contains the updated parameters.

       hErrorNode_out (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           -  node which caused the permissibility check to forbid the update.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorGraphExecUpdateFailure`
       * :py:obj:`~.hipGraphExecUpdateResult`:
               - Return code whether the graph update was performed.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph, hipGraphNode_t * hErrorNode_out, hipGraphExecUpdateResult * updateResult_out)


.. py:function:: hipGraphAddKernelNode(graph, pDependencies, numDependencies, pNodeParams)

   Creates a kernel execution node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to add the created node to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the dependencies of the kernel execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - The number of the dependencies.

       pNodeParams (:py:obj:`~.hipKernelNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the parameters of the kernel execution node.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorInvalidDeviceFunction`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddKernelNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipKernelNodeParams * pNodeParams)


.. py:function:: hipGraphKernelNodeGetParams(node)

   Gets kernel node's parameters.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - instance of the node to get parameters from.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipKernelNodeParams`:
               - pointer to the parameters

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams * pNodeParams)


.. py:function:: hipGraphKernelNodeSetParams(node, pNodeParams)

   Sets a kernel node's parameters.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       pNodeParams (:py:obj:`~.hipKernelNodeParams`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams * pNodeParams)


.. py:function:: hipGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)

   Sets the parameters for a kernel node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of the executable graph with the node.

       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       pNodeParams (:py:obj:`~.hipKernelNodeParams`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the kernel node parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipKernelNodeParams * pNodeParams)


.. py:function:: hipDrvGraphAddMemcpyNode(hGraph, dependencies, numDependencies, copyParams, ctx)

   Creates a memcpy node and adds it to a graph.

   Args:
       hGraph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to add the created node to.

       dependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies of the memcpy execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - The number of dependencies.

       copyParams (:py:obj:`~.HIP_MEMCPY3D`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the parameters for the memory copy.

       ctx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           - context related to current device.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t * phGraphNode, hipGraph_t hGraph, const hipGraphNode_t * dependencies, size_t numDependencies, const HIP_MEMCPY3D * copyParams, hipCtx_t ctx)


.. py:function:: hipGraphAddMemcpyNode(graph, pDependencies, numDependencies, pCopyParams)

   Creates a memcpy node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to add the created node to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies of the memcpy execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - The number of dependencies.

       pCopyParams (:py:obj:`~.hipMemcpy3DParms`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the parameters for the memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddMemcpyNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipMemcpy3DParms * pCopyParams)


.. py:function:: hipGraphMemcpyNodeGetParams(node)

   Gets a memcpy node's parameters.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - instance of the node to get parameters from.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipMemcpy3DParms`:
               - pointer to the parameters.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms * pNodeParams)


.. py:function:: hipGraphMemcpyNodeSetParams(node, pNodeParams)

   Sets a memcpy node's parameters.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - instance of the node to set parameters to.

       pNodeParams (:py:obj:`~.hipMemcpy3DParms`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms * pNodeParams)


.. py:function:: hipGraphKernelNodeSetAttribute(hNode, attr, value)

   Sets a node's attribute.

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       attr (:py:obj:`~.hipLaunchAttributeID`) -- *IN*:
           - The attribute type to be set.

       value (:py:obj:`~.hipLaunchAttributeValue`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipLaunchAttributeID attr, const hipLaunchAttributeValue * value)


.. py:function:: hipGraphKernelNodeGetAttribute(hNode, attr, value)

   Gets a node's attribute.

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       attr (:py:obj:`~.hipLaunchAttributeID`) -- *IN*:
           - The attribute type to be set.

       value (:py:obj:`~.hipLaunchAttributeValue`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipLaunchAttributeID attr, hipLaunchAttributeValue * value)


.. py:function:: hipGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)

   Sets the parameters of a memcpy node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of the executable graph with the node.

       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       pNodeParams (:py:obj:`~.hipMemcpy3DParms`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the kernel node parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipMemcpy3DParms * pNodeParams)


.. py:function:: hipGraphAddMemcpyNode1D(graph, pDependencies, numDependencies, dst, src, count, kind)

   Creates a 1D memcpy node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to add the created node to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies of the memcpy execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - The number of dependencies.

       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the destination.

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the source.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of the memory to copy.

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           - Type of memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, hipMemcpyKind kind)


.. py:function:: hipGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)

   Sets a memcpy node's parameters to perform a 1-dimensional copy.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the destination.

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the source.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of the memory to copy.

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           - Type of memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void * dst, const void * src, size_t count, hipMemcpyKind kind)


.. py:function:: hipGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)

   Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional
   copy.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of the executable graph with the node.

       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the destination.

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the source.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of the memory to copy.

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           - Type of memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node, void * dst, const void * src, size_t count, hipMemcpyKind kind)


.. py:function:: hipGraphAddMemcpyNodeFromSymbol(graph, pDependencies, numDependencies, dst, symbol, count, offset, kind)

   Creates a memcpy node to copy from a symbol on the device and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to add the created node to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies of the memcpy execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of the dependencies.

       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the destination.

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Device symbol address.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of the memory to copy.

       offset (:py:obj:`~.int`) -- *IN*:
           - Offset from start of symbol in bytes.

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           - Type of memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, hipMemcpyKind kind)


.. py:function:: hipGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind)

   Sets a memcpy node's parameters to copy from a symbol on the device.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the destination.

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Device symbol address.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of the memory to copy.

       offset (:py:obj:`~.int`) -- *IN*:
           - Offset from start of symbol in bytes.

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           - Type of memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, hipMemcpyKind kind)


.. py:function:: hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count, offset, kind)

   Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the
   * device.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of the executable graph with the node.

       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the destination.

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Device symbol address.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of the memory to copy.

       offset (:py:obj:`~.int`) -- *IN*:
           - Offset from start of symbol in bytes.

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           - Type of memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, hipMemcpyKind kind)


.. py:function:: hipGraphAddMemcpyNodeToSymbol(graph, pDependencies, numDependencies, symbol, src, count, offset, kind)

   Creates a memcpy node to copy to a symbol on the device and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of graph to add the created node to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies on the memcpy execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies.

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Device symbol address.

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the src.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of the memory to copy.

       offset (:py:obj:`~.int`) -- *IN*:
           - Offset from start of symbol in bytes.

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           - Type of memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, hipMemcpyKind kind)


.. py:function:: hipGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind)

   Sets a memcpy node's parameters to copy to a symbol on the device.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Device symbol address.

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the src.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of the memory to copy.

       offset (:py:obj:`~.int`) -- *IN*:
           - Offset from start of symbol in bytes.

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           - Type of memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, hipMemcpyKind kind)


.. py:function:: hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count, offset, kind)

   Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the
   device.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of the executable graph with the node.

       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Device symbol address.

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to memory address of the src.

       count (:py:obj:`~.int`) -- *IN*:
           - Size of the memory to copy.

       offset (:py:obj:`~.int`) -- *IN*:
           - Offset from start of symbol in bytes.

       kind (:py:obj:`~.hipMemcpyKind`) -- *IN*:
           - Type of memory copy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, hipMemcpyKind kind)


.. py:function:: hipGraphAddMemsetNode(graph, pDependencies, numDependencies, pMemsetParams)

   Creates a memset node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph to add the created node to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies on the memset execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies.

       pMemsetParams (:py:obj:`~.hipMemsetParams`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the parameters for the memory set.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddMemsetNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipMemsetParams * pMemsetParams)


.. py:function:: hipGraphMemsetNodeGetParams(node)

   Gets a memset node's parameters.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to get parameters of.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipMemsetParams`:
               - Pointer to the parameters.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams * pNodeParams)


.. py:function:: hipGraphMemsetNodeSetParams(node, pNodeParams)

   Sets a memset node's parameters.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       pNodeParams (:py:obj:`~.hipMemsetParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams * pNodeParams)


.. py:function:: hipGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)

   Sets the parameters for a memset node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of the executable graph with the node.

       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       pNodeParams (:py:obj:`~.hipMemsetParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipMemsetParams * pNodeParams)


.. py:function:: hipGraphAddHostNode(graph, pDependencies, numDependencies, pNodeParams)

   Creates a host execution node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph to add the created node to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies of the memset execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies.

       pNodeParams (:py:obj:`~.hipHostNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddHostNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipHostNodeParams * pNodeParams)


.. py:function:: hipGraphHostNodeGetParams(node)

   Returns a host node's parameters.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to get parameters of.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipHostNodeParams`:
               - Pointer to the parameters.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams * pNodeParams)


.. py:function:: hipGraphHostNodeSetParams(node, pNodeParams)

   Sets a host node's parameters.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       pNodeParams (:py:obj:`~.hipHostNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams * pNodeParams)


.. py:function:: hipGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)

   Sets the parameters for a host node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - Instance of the executable graph with the node.

       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set parameters of.

       pNodeParams (:py:obj:`~.hipHostNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipHostNodeParams * pNodeParams)


.. py:function:: hipGraphAddChildGraphNode(graph, pDependencies, numDependencies, childGraph)

   Creates a child graph node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph to add the created node.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies of the memset execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies.

       childGraph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Graph to clone into this node

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddChildGraphNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipGraph_t childGraph)


.. py:function:: hipGraphChildGraphNodeGetGraph(node)

   Gets a handle to the embedded graph of a child graph node.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to get child graph of.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipGraph`:
               - Pointer to get the graph.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t * pGraph)


.. py:function:: hipGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)

   Updates node parameters in the child graph node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - instance of the executable graph with the node.

       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - node from the graph which was used to instantiate graphExec.

       childGraph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - child graph with updated parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipGraph_t childGraph)


.. py:function:: hipGraphAddEmptyNode(graph, pDependencies, numDependencies)

   Creates an empty node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph the node is added to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the node dependencies.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddEmptyNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies)


.. py:function:: hipGraphAddEventRecordNode(graph, pDependencies, numDependencies, event)

   Creates an event record node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph the node is added to.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the node dependencies.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies.

       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           - Event of the node.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddEventRecordNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipEvent_t event)


.. py:function:: hipGraphEventRecordNodeGetEvent(node)

   Returns the event associated with an event record node.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           -  Instance of the node to get event of.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipEvent_t`:
               - Pointer to return the event.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t * event_out)


.. py:function:: hipGraphEventRecordNodeSetEvent(node, event)

   Sets an event record node's event.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set event to.

       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the event.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event)


.. py:function:: hipGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)

   Sets the event for an event record node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - instance of the executable graph with the node.

       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - node from the graph which was used to instantiate graphExec.

       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           - pointer to the event.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event)


.. py:function:: hipGraphAddEventWaitNode(graph, pDependencies, numDependencies, event)

   Creates an event wait node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph the node to be added.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the node dependencies.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - Number of dependencies.

       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           - Event for the node.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to graph node that is created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddEventWaitNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipEvent_t event)


.. py:function:: hipGraphEventWaitNodeGetEvent(node)

   Returns the event associated with an event wait node.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           -  Instance of the node to get event of.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.ihipEvent_t`:
               - Pointer to return the event.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t * event_out)


.. py:function:: hipGraphEventWaitNodeSetEvent(node, event)

   Sets an event wait node's event.

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Instance of the node to set event of.

       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the event.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event)


.. py:function:: hipGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)

   Sets the event for an event record node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - instance of the executable graph with the node.

       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - node from the graph which was used to instantiate graphExec.

       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`) -- *IN*:
           - pointer to the event.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event)


.. py:function:: hipGraphAddMemAllocNode(graph, pDependencies, numDependencies, pNodeParams)

   Creates a memory allocation node and adds it to a graph

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph node to be added

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Const pointer to the node dependencies

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - The number of dependencies

       pNodeParams (:py:obj:`~.hipMemAllocNodeParams`/:py:obj:`~.object`) -- *IN,OUT*:
           - Node parameters for memory allocation, returns a pointer to the
           allocated memory.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to the graph node to create and add to the graph

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddMemAllocNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipMemAllocNodeParams * pNodeParams)


.. py:function:: hipGraphMemAllocNodeGetParams(node)

   Returns parameters for memory allocation node

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Memory allocation node to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipMemAllocNodeParams`:
               - Parameters for the specified memory allocation node

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams * pNodeParams)


.. py:function:: hipGraphAddMemFreeNode(graph, pDependencies, numDependencies, dev_ptr)

   Creates a memory free node and adds it to a graph

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph node to be added

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Const pointer to the node dependencies

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - The number of dependencies

       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the memory to be freed

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to the graph node to create and add to the graph

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddMemFreeNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, void * dev_ptr)


.. py:function:: hipGraphMemFreeNodeGetParams(node, dev_ptr)

   Returns parameters for memory free node

   Args:
       node (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Memory free node to query

       dev_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           - Device pointer of the specified memory free node

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void * dev_ptr)


.. py:function:: hipDeviceGetGraphMemAttribute(device, attr, value)

   Get the mem attribute for graphs.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           - Device to get attributes from

       attr (:py:obj:`~.hipGraphMemAttributeType`) -- *IN*:
           - Attribute type to be queried

       value (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           - Value of the queried attribute

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void * value)


.. py:function:: hipDeviceSetGraphMemAttribute(device, attr, value)

   Set the mem attribute for graphs.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           - Device to set attribute of.

       attr (:py:obj:`~.hipGraphMemAttributeType`) -- *IN*:
           - Attribute type to be set.

       value (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Value of the attribute.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void * value)


.. py:function:: hipDeviceGraphMemTrim(device)

   Free unused memory reserved for graphs on a specific device and return it back to the OS.

   Args:
       device (:py:obj:`~.int`) -- *IN*:
           - Device for which memory should be trimmed

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidDevice`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDeviceGraphMemTrim(int device)


.. py:function:: hipUserObjectCreate(ptr, destroy, initialRefcount, flags)

   Create an instance of userObject to manage lifetime of a resource.

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - pointer to pass to destroy function.

       destroy (:py:obj:`~.hipHostFn_t`/:py:obj:`~.object`) -- *IN*:
           - destroy callback to remove resource.

       initialRefcount (:py:obj:`~.int`) -- *IN*:
           - reference to resource.

       flags (:py:obj:`~.int`) -- *IN*:
           - flags passed to API.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipUserObject`:
               - pointer to instace of userobj.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipUserObjectCreate(hipUserObject_t * object_out, void * ptr, hipHostFn_t destroy, unsigned int initialRefcount, unsigned int flags)


.. py:function:: hipUserObjectRelease(object, count)

   Release number of references to resource.

   Args:
       object (:py:obj:`~.hipUserObject`/:py:obj:`~.object`) -- *IN*:
           - pointer to instace of userobj.

       count (:py:obj:`~.int`) -- *IN*:
           - reference to resource to be retained.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count)


.. py:function:: hipUserObjectRetain(object, count)

   Retain number of references to resource.

   Args:
       object (:py:obj:`~.hipUserObject`/:py:obj:`~.object`) -- *IN*:
           - pointer to instace of userobj.

       count (:py:obj:`~.int`) -- *IN*:
           - reference to resource to be retained.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count)


.. py:function:: hipGraphRetainUserObject(graph, object, count, flags)

   Retain user object for graphs.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - pointer to graph to retain the user object for.

       object (:py:obj:`~.hipUserObject`/:py:obj:`~.object`) -- *IN*:
           - pointer to instace of userobj.

       count (:py:obj:`~.int`) -- *IN*:
           - reference to resource to be retained.

       flags (:py:obj:`~.int`) -- *IN*:
           - flags passed to API.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count, unsigned int flags)


.. py:function:: hipGraphReleaseUserObject(graph, object, count)

   Release user object from graphs.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - pointer to graph to retain the user object for.

       object (:py:obj:`~.hipUserObject`/:py:obj:`~.object`) -- *IN*:
           - pointer to instace of userobj.

       count (:py:obj:`~.int`) -- *IN*:
           - reference to resource to be retained.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count)


.. py:function:: hipGraphDebugDotPrint(graph, path, flags)

   Write a DOT file describing graph structure.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - graph object for which DOT file has to be generated.

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           - path to write the DOT file.

       flags (:py:obj:`~.int`) -- *IN*:
           - Flags from hipGraphDebugDotFlags to get additional node information.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorOperatingSystem`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char * path, unsigned int flags)


.. py:function:: hipGraphKernelNodeCopyAttributes(hSrc)

   Copies attributes from source node to destination node.

   Copies attributes from source node to destination node.
   Both node must have the same context.

   Args:
       hSrc (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Source node.
           For list of attributes see ::hipKernelNodeAttrID.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidContext`
       * :py:obj:`~.hipGraphNode`:
               - Destination node.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst)


.. py:function:: hipGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)

   Enables or disables the specified node in the given graphExec

   Sets hNode to be either enabled or disabled. Disabled nodes are functionally equivalent
   to empty nodes until they are reenabled. Existing node parameters are not affected by
   disabling/enabling the node.

   The node is identified by the corresponding hNode in the non-executable graph, from which the
   executable graph was instantiated.

   hNode must not have been removed from the original graph.

   Note:
       Currently only kernel, memset and memcpy nodes are supported.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - The executable graph in which to set the specified node.

       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node from the graph from which graphExec was instantiated.

       isEnabled (:py:obj:`~.int`) -- *IN*:
           - Node is enabled if != 0, otherwise the node is disabled.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`,

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int isEnabled)


.. py:function:: hipGraphNodeGetEnabled(hGraphExec, hNode)

   Query whether a node in the given graphExec is enabled

   Sets isEnabled to 1 if hNode is enabled, or 0 if it is disabled.

   The node is identified by the corresponding node in the non-executable graph, from which the
   executable graph was instantiated.

   hNode must not have been removed from the original graph.

   Note:
       Currently only kernel, memset and memcpy nodes are supported.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - The executable graph in which to set the specified node.

       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node from the graph from which graphExec was instantiated.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.int`:
               - Location to return the enabled status of the node.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int * isEnabled)


.. py:function:: hipGraphAddExternalSemaphoresWaitNode(graph, pDependencies, numDependencies, nodeParams)

   Creates a external semaphor wait node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - instance of the graph to add the created node.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies on the memset execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - the number of the dependencies.

       nodeParams (:py:obj:`~.hipExternalSemaphoreWaitNodeParams`/:py:obj:`~.object`) -- *IN*:
           -pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - pointer to the graph node to create.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddExternalSemaphoresWaitNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipExternalSemaphoreWaitNodeParams * nodeParams)


.. py:function:: hipGraphAddExternalSemaphoresSignalNode(graph, pDependencies, numDependencies, nodeParams)

   Creates a external semaphor signal node and adds it to a graph.

   Args:
       graph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - instance of the graph to add the created node.

       pDependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies on the memset execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - the number of the dependencies.

       nodeParams (:py:obj:`~.hipExternalSemaphoreSignalNodeParams`/:py:obj:`~.object`) -- *IN*:
           -pointer to the parameters.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - pointer to the graph node to create.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphAddExternalSemaphoresSignalNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipExternalSemaphoreSignalNodeParams * nodeParams)


.. py:function:: hipGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)

   Updates node parameters in the external semaphore signal node.

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node from the graph from which graphExec was instantiated.

       nodeParams (:py:obj:`~.hipExternalSemaphoreSignalNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the params to be set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams * nodeParams)


.. py:function:: hipGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)

   Updates node parameters in the external semaphore wait node.

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node from the graph from which graphExec was instantiated.

       nodeParams (:py:obj:`~.hipExternalSemaphoreWaitNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the params to be set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams * nodeParams)


.. py:function:: hipGraphExternalSemaphoresSignalNodeGetParams(hNode)

   Returns external semaphore signal node params.

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node from the graph from which graphExec was instantiated.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipExternalSemaphoreSignalNodeParams`:
               - Pointer to params.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams * params_out)


.. py:function:: hipGraphExternalSemaphoresWaitNodeGetParams(hNode)

   Returns external semaphore wait node params.

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node from the graph from which graphExec was instantiated.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipExternalSemaphoreWaitNodeParams`:
               - Pointer to params.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams * params_out)


.. py:function:: hipGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)

   Updates node parameters in the external semaphore signal node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - The executable graph in which to set the specified node.

       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node from the graph from which graphExec was instantiated.

       nodeParams (:py:obj:`~.hipExternalSemaphoreSignalNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the params to be set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams * nodeParams)


.. py:function:: hipGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)

   Updates node parameters in the external semaphore wait node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - The executable graph in which to set the specified node.

       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - Node from the graph from which graphExec was instantiated.

       nodeParams (:py:obj:`~.hipExternalSemaphoreWaitNodeParams`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the params to be set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams * nodeParams)


.. py:function:: hipDrvGraphMemcpyNodeGetParams(hNode)

   Gets a memcpy node's parameters.

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - instance of the node to get parameters from.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.HIP_MEMCPY3D`:
               - pointer to the parameters.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t hNode, HIP_MEMCPY3D * nodeParams)


.. py:function:: hipDrvGraphMemcpyNodeSetParams(hNode)

   Sets a memcpy node's parameters.

   Args:
       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - instance of the node to Set parameters for.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.HIP_MEMCPY3D`:
               - pointer to the parameters.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvGraphMemcpyNodeSetParams(hipGraphNode_t hNode, const HIP_MEMCPY3D * nodeParams)


.. py:function:: hipDrvGraphAddMemsetNode(hGraph, dependencies, numDependencies, memsetParams, ctx)

   Creates a memset node and adds it to a graph.

   Args:
       hGraph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - instance of graph to add the created node to.

       dependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the dependencies on the memset execution node.

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - number of the dependencies.

       memsetParams (:py:obj:`~.hipMemsetParams`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the parameters for the memory set.

       ctx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           - cotext related to current device.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - pointer to graph node to create.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t * phGraphNode, hipGraph_t hGraph, const hipGraphNode_t * dependencies, size_t numDependencies, const hipMemsetParams * memsetParams, hipCtx_t ctx)


.. py:function:: hipDrvGraphAddMemFreeNode(hGraph, dependencies, numDependencies, dptr)

   Creates a memory free node and adds it to a graph

   Args:
       hGraph (:py:obj:`~.ihipGraph`/:py:obj:`~.object`) -- *IN*:
           - Instance of the graph the node to be added

       dependencies (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Const pointer to the node dependencies

       numDependencies (:py:obj:`~.int`) -- *IN*:
           - The number of dependencies

       dptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer to the memory to be freed

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipGraphNode`:
               - Pointer to the graph node to create and add to the graph

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvGraphAddMemFreeNode(hipGraphNode_t * phGraphNode, hipGraph_t hGraph, const hipGraphNode_t * dependencies, size_t numDependencies, hipDeviceptr_t dptr)


.. py:function:: hipDrvGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx)

   Sets the parameters for a memcpy node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - instance of the executable graph with the node.

       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - instance of the node to set parameters to.

       copyParams (:py:obj:`~.HIP_MEMCPY3D`/:py:obj:`~.object`) -- *IN*:
           - const pointer to the memcpy node params.

       ctx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           - cotext related to current device.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const HIP_MEMCPY3D * copyParams, hipCtx_t ctx)


.. py:function:: hipDrvGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx)

   Sets the parameters for a memset node in the given graphExec.

   Args:
       hGraphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`) -- *IN*:
           - instance of the executable graph with the node.

       hNode (:py:obj:`~.hipGraphNode`/:py:obj:`~.object`) -- *IN*:
           - instance of the node to set parameters to.

       memsetParams (:py:obj:`~.hipMemsetParams`/:py:obj:`~.object`) -- *IN*:
           - pointer to the parameters.

       ctx (:py:obj:`~.ihipCtx_t`/:py:obj:`~.object`) -- *IN*:
           - cotext related to current device.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDrvGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipMemsetParams * memsetParams, hipCtx_t ctx)


.. py:function:: hipMemAddressFree(devPtr, size)

   Frees an address range reservation made via hipMemAddressReserve

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       devPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - starting address of the range.

       size (:py:obj:`~.int`) -- *IN*:
           - size of the range.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemAddressFree(void * devPtr, size_t size)


.. py:function:: hipMemAddressReserve(size, alignment, addr, flags)

   Reserves an address range

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           - size of the reservation.

       alignment (:py:obj:`~.int`) -- *IN*:
           - alignment of the address.

       addr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - requested starting address of the range.

       flags (:py:obj:`~.int`) -- *IN*:
           - currently unused, must be zero.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               - starting address of the reserved range.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemAddressReserve(void ** ptr, size_t size, size_t alignment, void * addr, unsigned long long flags)


.. py:function:: hipMemCreate(size, prop, flags)

   Creates a memory handle for the allocation described by the properties and given size

   This API creates a memory allocation on the target device specified through the prop structure.
   The prop allocation type must be specified as either :py:obj:`~.hipMemAllocationTypePinned` or
   :py:obj:`~.hipMemAllocationTypeUncached`.
   The prop location type must be specified as :py:obj:`~.hipMemLocationTypeDevice` or :py:obj:`~.hipMemLocationTypeHost`.
   Any other value results in :py:obj:`~.hipErrorInvalidValue`.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       size (:py:obj:`~.int`) -- *IN*:
           - size of the allocation.

       prop (:py:obj:`~.hipMemAllocationProp`/:py:obj:`~.object`) -- *IN*:
           - properties of the allocation.

       flags (:py:obj:`~.int`) -- *IN*:
           - currently unused, must be zero.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.ihipMemGenericAllocationHandle`:
               - value of the returned handle.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemCreate(hipMemGenericAllocationHandle_t * handle, size_t size, const hipMemAllocationProp * prop, unsigned long long flags)


.. py:function:: hipMemExportToShareableHandle(shareableHandle, handle, handleType, flags)

   Exports an allocation to a requested shareable handle type.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       shareableHandle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           - value of the returned handle.

       handle (:py:obj:`~.ihipMemGenericAllocationHandle`/:py:obj:`~.object`) -- *IN*:
           - handle to share.

       handleType (:py:obj:`~.hipMemAllocationHandleType`) -- *IN*:
           - type of the shareable handle.

       flags (:py:obj:`~.int`) -- *IN*:
           - currently unused, must be zero.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemExportToShareableHandle(void * shareableHandle, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType handleType, unsigned long long flags)


.. py:function:: hipMemGetAccess(location, ptr)

   Get the access flags set for the given location and ptr.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       location (:py:obj:`~.hipMemLocation`/:py:obj:`~.object`) -- *IN*:
           - target location.

       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - address to check the access flags.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               - flags for this location.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemGetAccess(unsigned long long * flags, const hipMemLocation * location, void * ptr)


.. py:function:: hipMemGetAllocationGranularity(prop, option)

   Calculates either the minimal or recommended granularity.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       prop (:py:obj:`~.hipMemAllocationProp`/:py:obj:`~.object`) -- *IN*:
           - location properties.

       option (:py:obj:`~.hipMemAllocationGranularity_flags`) -- *IN*:
           - determines which granularity to return.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.int`:
               - returned granularity.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemGetAllocationGranularity(size_t * granularity, const hipMemAllocationProp * prop, hipMemAllocationGranularity_flags option)


.. py:function:: hipMemGetAllocationPropertiesFromHandle(handle)

   Retrieve the property structure of the given handle.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       handle (:py:obj:`~.ihipMemGenericAllocationHandle`/:py:obj:`~.object`) -- *IN*:
           - handle to perform the query on.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.hipMemAllocationProp`:
               - properties of the given handle.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp * prop, hipMemGenericAllocationHandle_t handle)


.. py:function:: hipMemImportFromShareableHandle(osHandle, shHandleType)

   Imports an allocation from a requested shareable handle type.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       osHandle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - shareable handle representing the memory allocation.

       shHandleType (:py:obj:`~.hipMemAllocationHandleType`) -- *IN*:
           - handle type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.ihipMemGenericAllocationHandle`:
               - returned value.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t * handle, void * osHandle, hipMemAllocationHandleType shHandleType)


.. py:function:: hipMemMap(ptr, size, offset, handle, flags)

   Maps an allocation handle to a reserved virtual address range.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - address where the memory will be mapped.

       size (:py:obj:`~.int`) -- *IN*:
           - size of the mapping.

       offset (:py:obj:`~.int`) -- *IN*:
           - offset into the memory, currently must be zero.

       handle (:py:obj:`~.ihipMemGenericAllocationHandle`/:py:obj:`~.object`) -- *IN*:
           - memory allocation to be mapped.

       flags (:py:obj:`~.int`) -- *IN*:
           - currently unused, must be zero.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemMap(void * ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle, unsigned long long flags)


.. py:function:: hipMemMapArrayAsync(mapInfoList, count, stream)

   Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.

   Args:
       mapInfoList (:py:obj:`~.hipArrayMapInfo`/:py:obj:`~.object`) -- *IN*:
           - list of hipArrayMapInfo.

       count (:py:obj:`~.int`) -- *IN*:
           - number of hipArrayMapInfo in mapInfoList.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - stream identifier for the stream to use for map or unmap operations.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemMapArrayAsync(hipArrayMapInfo * mapInfoList, unsigned int count, hipStream_t stream)


.. py:function:: hipMemRelease(handle)

   Release a memory handle representing a memory allocation which was previously allocated
   through hipMemCreate.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       handle (:py:obj:`~.ihipMemGenericAllocationHandle`/:py:obj:`~.object`) -- *IN*:
           - handle of the memory allocation.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle)


.. py:function:: hipMemRetainAllocationHandle(addr)

   Returns the allocation handle of the backing memory allocation given the address.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       addr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - address to look up.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`
       * :py:obj:`~.ihipMemGenericAllocationHandle`:
               - handle representing addr.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t * handle, void * addr)


.. py:function:: hipMemSetAccess(ptr, size, desc, count)

   Set the access flags for each location specified in desc for the given virtual address
   range.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - starting address of the virtual address range.

       size (:py:obj:`~.int`) -- *IN*:
           - size of the range.

       desc (:py:obj:`~.hipMemAccessDesc`/:py:obj:`~.object`) -- *IN*:
           - array of hipMemAccessDesc.

       count (:py:obj:`~.int`) -- *IN*:
           - number of hipMemAccessDesc in desc.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemSetAccess(void * ptr, size_t size, const hipMemAccessDesc * desc, size_t count)


.. py:function:: hipMemUnmap(ptr, size)

   Unmap memory allocation of a given address range.

   Note:
       This API is implemented on Linux and is under development on Microsoft Windows.

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - starting address of the range to unmap.

       size (:py:obj:`~.int`) -- *IN*:
           - size of the virtual address range.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorNotSupported`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemUnmap(void * ptr, size_t size)


.. py:function:: hipGraphicsMapResources(count, resources, stream)

   Maps a graphics resource for access.

   Args:
       count (:py:obj:`~.int`) -- *IN*:
           - Number of resources to map.

       resources (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer of resources to map.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream for synchronization.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorUnknown`, :py:obj:`~.hipErrorInvalidResourceHandle`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t * resources, hipStream_t stream)


.. py:function:: hipGraphicsSubResourceGetMappedArray(resource, arrayIndex, mipLevel)

   Get an array through which to access a subresource of a mapped graphics resource.

   Note:
       In this API, the value of arrayIndex higher than zero is currently not supported.

   Args:
       resource (:py:obj:`~._hipGraphicsResource`/:py:obj:`~.object`) -- *IN*:
           - Mapped resource to access.

       arrayIndex (:py:obj:`~.int`) -- *IN*:
           - Array index for the subresource to access.

       mipLevel (:py:obj:`~.int`) -- *IN*:
           - Mipmap level for the subresource to access.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.hipArray`:
               - Pointer of array through which a subresource of resource may be accessed.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t * array, hipGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)


.. py:function:: hipGraphicsResourceGetMappedPointer(resource)

   Gets device accessible address of a graphics resource.

   Args:
       resource (:py:obj:`~._hipGraphicsResource`/:py:obj:`~.object`) -- *IN*:
           - Mapped resource to access.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               - Pointer of device through which graphic resource may be accessed.
       * :py:obj:`~.int`:
               - Size of the buffer accessible from devPtr.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, hipGraphicsResource_t resource)


.. py:function:: hipGraphicsUnmapResources(count, resources, stream)

   Unmaps graphics resources.

   Args:
       count (:py:obj:`~.int`) -- *IN*:
           - Number of resources to unmap.

       resources (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Pointer of resources to unmap.

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`) -- *IN*:
           - Stream for synchronization.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`, :py:obj:`~.hipErrorUnknown`, :py:obj:`~.hipErrorContextIsDestroyed`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t * resources, hipStream_t stream)


.. py:function:: hipGraphicsUnregisterResource(resource)

   Unregisters a graphics resource.

   Args:
       resource (:py:obj:`~._hipGraphicsResource`/:py:obj:`~.object`) -- *IN*:
           - Graphics resources to unregister.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource)


.. py:function:: hipCreateSurfaceObject(pResDesc)

   Create a surface object.

   Args:
       pResDesc (:py:obj:`~.hipResourceDesc`/:py:obj:`~.object`) -- *IN*:
           Pointer of suface object descriptor.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`
       * :py:obj:`~.__hip_surface`:
               Pointer of surface object to be created.

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipCreateSurfaceObject(hipSurfaceObject_t * pSurfObject, const hipResourceDesc * pResDesc)


.. py:function:: hipDestroySurfaceObject(surfaceObject)

   Destroy a surface object.

   Args:
       surfaceObject (:py:obj:`~.__hip_surface`/:py:obj:`~.object`) -- *IN*:
           Surface object to be destroyed.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject)


.. py:function:: hipExtEnableLogging()

   Enable HIP runtime logging.

   This function enables the HIP runtime logging mechanism, allowing diagnostic
   and trace information to be captured during HIP API execution.

   See:
       :py:obj:`~.hipExtDisableLogging`, :py:obj:`~.hipExtSetLoggingParams`

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtEnableLogging()


.. py:function:: hipExtDisableLogging()

   Disable HIP runtime logging.

   This function disables the HIP runtime logging mechanism, stopping the capture
   of diagnostic and trace information during HIP API execution.

   See:
       :py:obj:`~.hipExtEnableLogging`, :py:obj:`~.hipExtSetLoggingParams`

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtDisableLogging()


.. py:function:: hipExtSetLoggingParams(log_level, log_size, log_mask)

   Set HIP runtime logging parameters.

   This function configures the logging behavior of the HIP runtime, including
   the verbosity level, buffer size, and which components to log.

   See:
       :py:obj:`~.hipExtEnableLogging`, :py:obj:`~.hipExtDisableLogging`

   Args:
       log_level (:py:obj:`~.int`) -- *IN*:
           The logging verbosity level. Higher values produce more detailed output.

       log_size (:py:obj:`~.int`) -- *IN*:
           Reserved for future use. Currently not implemented.

       log_mask (:py:obj:`~.int`) -- *IN*:
           A bitmask specifying which HIP runtime components to log.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: py:obj:`~.hipSuccess`, :py:obj:`~.hipErrorInvalidValue`

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipExtSetLoggingParams(size_t log_level, size_t log_size, size_t log_mask)


.. py:function:: hipMemcpy_spt(dst, src, sizeBytes, kind)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sizeBytes (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy_spt(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind)


.. py:function:: hipMemcpyToSymbol_spt(symbol, src, sizeBytes, offset, kind)

   (No short description, might be part of a group.)

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sizeBytes (:py:obj:`~.int`):
           (undocumented)

       offset (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyToSymbol_spt(const void * symbol, const void * src, size_t sizeBytes, size_t offset, hipMemcpyKind kind)


.. py:function:: hipMemcpyFromSymbol_spt(dst, symbol, sizeBytes, offset, kind)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sizeBytes (:py:obj:`~.int`):
           (undocumented)

       offset (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyFromSymbol_spt(void * dst, const void * symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind)


.. py:function:: hipMemcpy2D_spt(dst, dpitch, src, spitch, width, height, kind)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       dpitch (:py:obj:`~.int`):
           (undocumented)

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       spitch (:py:obj:`~.int`):
           (undocumented)

       width (:py:obj:`~.int`):
           (undocumented)

       height (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2D_spt(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind)


.. py:function:: hipMemcpy2DFromArray_spt(dst, dpitch, src, wOffset, hOffset, width, height, kind)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       dpitch (:py:obj:`~.int`):
           (undocumented)

       src (:py:obj:`~.hipArray`/:py:obj:`~.object`):
           (undocumented)

       wOffset (:py:obj:`~.int`):
           (undocumented)

       hOffset (:py:obj:`~.int`):
           (undocumented)

       width (:py:obj:`~.int`):
           (undocumented)

       height (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DFromArray_spt(void * dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind)


.. py:function:: hipMemcpy3D_spt(p)

   (No short description, might be part of a group.)

   Args:
       p (:py:obj:`~.hipMemcpy3DParms`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms * p)


.. py:function:: hipMemset_spt(dst, value, sizeBytes)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       value (:py:obj:`~.int`):
           (undocumented)

       sizeBytes (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset_spt(void * dst, int value, size_t sizeBytes)


.. py:function:: hipMemsetAsync_spt(dst, value, sizeBytes, stream)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       value (:py:obj:`~.int`):
           (undocumented)

       sizeBytes (:py:obj:`~.int`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemsetAsync_spt(void * dst, int value, size_t sizeBytes, hipStream_t stream)


.. py:function:: hipMemset2D_spt(dst, pitch, value, width, height)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       pitch (:py:obj:`~.int`):
           (undocumented)

       value (:py:obj:`~.int`):
           (undocumented)

       width (:py:obj:`~.int`):
           (undocumented)

       height (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset2D_spt(void * dst, size_t pitch, int value, size_t width, size_t height)


.. py:function:: hipMemset2DAsync_spt(dst, pitch, value, width, height, stream)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       pitch (:py:obj:`~.int`):
           (undocumented)

       value (:py:obj:`~.int`):
           (undocumented)

       width (:py:obj:`~.int`):
           (undocumented)

       height (:py:obj:`~.int`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset2DAsync_spt(void * dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream)


.. py:function:: hipMemset3DAsync_spt(pitchedDevPtr, value, extent, stream)

   (No short description, might be part of a group.)

   Args:
       pitchedDevPtr (:py:obj:`~.hipPitchedPtr`):
           (undocumented)

       value (:py:obj:`~.int`):
           (undocumented)

       extent (:py:obj:`~.hipExtent`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent, hipStream_t stream)


.. py:function:: hipMemset3D_spt(pitchedDevPtr, value, extent)

   (No short description, might be part of a group.)

   Args:
       pitchedDevPtr (:py:obj:`~.hipPitchedPtr`):
           (undocumented)

       value (:py:obj:`~.int`):
           (undocumented)

       extent (:py:obj:`~.hipExtent`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent)


.. py:function:: hipMemcpyAsync_spt(dst, src, sizeBytes, kind, stream)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sizeBytes (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyAsync_spt(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpy3DAsync_spt(p, stream)

   (No short description, might be part of a group.)

   Args:
       p (:py:obj:`~.hipMemcpy3DParms`/:py:obj:`~.object`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy3DAsync_spt(const hipMemcpy3DParms * p, hipStream_t stream)


.. py:function:: hipMemcpy2DAsync_spt(dst, dpitch, src, spitch, width, height, kind, stream)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       dpitch (:py:obj:`~.int`):
           (undocumented)

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       spitch (:py:obj:`~.int`):
           (undocumented)

       width (:py:obj:`~.int`):
           (undocumented)

       height (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DAsync_spt(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpyFromSymbolAsync_spt(dst, symbol, sizeBytes, offset, kind, stream)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sizeBytes (:py:obj:`~.int`):
           (undocumented)

       offset (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyFromSymbolAsync_spt(void * dst, const void * symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpyToSymbolAsync_spt(symbol, src, sizeBytes, offset, kind, stream)

   (No short description, might be part of a group.)

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sizeBytes (:py:obj:`~.int`):
           (undocumented)

       offset (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyToSymbolAsync_spt(const void * symbol, const void * src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpyFromArray_spt(dst, src, wOffsetSrc, hOffset, count, kind)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       src (:py:obj:`~.hipArray`/:py:obj:`~.object`):
           (undocumented)

       wOffsetSrc (:py:obj:`~.int`):
           (undocumented)

       hOffset (:py:obj:`~.int`):
           (undocumented)

       count (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpyFromArray_spt(void * dst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffset, size_t count, hipMemcpyKind kind)


.. py:function:: hipMemcpy2DToArray_spt(dst, wOffset, hOffset, src, spitch, width, height, kind)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.hipArray`/:py:obj:`~.object`):
           (undocumented)

       wOffset (:py:obj:`~.int`):
           (undocumented)

       hOffset (:py:obj:`~.int`):
           (undocumented)

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       spitch (:py:obj:`~.int`):
           (undocumented)

       width (:py:obj:`~.int`):
           (undocumented)

       height (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind)


.. py:function:: hipMemcpy2DFromArrayAsync_spt(dst, dpitch, src, wOffsetSrc, hOffsetSrc, width, height, kind, stream)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       dpitch (:py:obj:`~.int`):
           (undocumented)

       src (:py:obj:`~.hipArray`/:py:obj:`~.object`):
           (undocumented)

       wOffsetSrc (:py:obj:`~.int`):
           (undocumented)

       hOffsetSrc (:py:obj:`~.int`):
           (undocumented)

       width (:py:obj:`~.int`):
           (undocumented)

       height (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DFromArrayAsync_spt(void * dst, size_t dpitch, hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipMemcpy2DToArrayAsync_spt(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)

   (No short description, might be part of a group.)

   Args:
       dst (:py:obj:`~.hipArray`/:py:obj:`~.object`):
           (undocumented)

       wOffset (:py:obj:`~.int`):
           (undocumented)

       hOffset (:py:obj:`~.int`):
           (undocumented)

       src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       spitch (:py:obj:`~.int`):
           (undocumented)

       width (:py:obj:`~.int`):
           (undocumented)

       height (:py:obj:`~.int`):
           (undocumented)

       kind (:py:obj:`~.hipMemcpyKind`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipMemcpy2DToArrayAsync_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream)


.. py:function:: hipStreamQuery_spt(stream)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamQuery_spt(hipStream_t stream)


.. py:function:: hipStreamSynchronize_spt(stream)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamSynchronize_spt(hipStream_t stream)


.. py:function:: hipStreamGetPriority_spt(stream)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * priority (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetPriority_spt(hipStream_t stream, int * priority)


.. py:function:: hipStreamWaitEvent_spt(stream, event, flags)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`):
           (undocumented)

       flags (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags)


.. py:function:: hipStreamGetFlags_spt(stream)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * flags (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int * flags)


.. py:function:: hipStreamAddCallback_spt(stream, callback, userData, flags)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

       callback (:py:obj:`~.hipStreamCallback_t`/:py:obj:`~.object`):
           (undocumented)

       userData (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       flags (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamAddCallback_spt(hipStream_t stream, hipStreamCallback_t callback, void * userData, unsigned int flags)


.. py:function:: hipEventRecord_spt(event, stream)

   (No short description, might be part of a group.)

   Args:
       event (:py:obj:`~.ihipEvent_t`/:py:obj:`~.object`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream)


.. py:function:: hipLaunchCooperativeKernel_spt(f, gridDim, blockDim, sharedMemBytes, hStream)

   (No short description, might be part of a group.)

   Args:
       f (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       gridDim (:py:obj:`~.dim3`):
           (undocumented)

       blockDim (:py:obj:`~.dim3`):
           (undocumented)

       sharedMemBytes (:py:obj:`~.int`):
           (undocumented)

       hStream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * kernelParams (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLaunchCooperativeKernel_spt(const void * f, dim3 gridDim, dim3 blockDim, void ** kernelParams, uint32_t sharedMemBytes, hipStream_t hStream)


.. py:function:: hipLaunchKernel_spt(function_address, numBlocks, dimBlocks, sharedMemBytes, stream)

   (No short description, might be part of a group.)

   Args:
       function_address (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       numBlocks (:py:obj:`~.dim3`):
           (undocumented)

       dimBlocks (:py:obj:`~.dim3`):
           (undocumented)

       sharedMemBytes (:py:obj:`~.int`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLaunchKernel_spt(const void * function_address, dim3 numBlocks, dim3 dimBlocks, void ** args, size_t sharedMemBytes, hipStream_t stream)


.. py:function:: hipGraphLaunch_spt(graphExec, stream)

   (No short description, might be part of a group.)

   Args:
       graphExec (:py:obj:`~.hipGraphExec`/:py:obj:`~.object`):
           (undocumented)

       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec, hipStream_t stream)


.. py:function:: hipStreamBeginCapture_spt(stream, mode)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

       mode (:py:obj:`~.hipStreamCaptureMode`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamBeginCapture_spt(hipStream_t stream, hipStreamCaptureMode mode)


.. py:function:: hipStreamEndCapture_spt(stream)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * pGraph (:py:obj:`~.ihipGraph`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamEndCapture_spt(hipStream_t stream, hipGraph_t * pGraph)


.. py:function:: hipStreamIsCapturing_spt(stream)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * pCaptureStatus (:py:obj:`~.hipStreamCaptureStatus`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamIsCapturing_spt(hipStream_t stream, hipStreamCaptureStatus * pCaptureStatus)


.. py:function:: hipStreamGetCaptureInfo_spt(stream)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * pCaptureStatus (:py:obj:`~.hipStreamCaptureStatus`):
           (undocumented)
       * pId (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, hipStreamCaptureStatus * pCaptureStatus, unsigned long long * pId)


.. py:function:: hipStreamGetCaptureInfo_v2_spt(stream, dependencies_out)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

       dependencies_out (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 5 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * captureStatus_out (:py:obj:`~.hipStreamCaptureStatus`):
           (undocumented)
       * id_out (:py:obj:`~.int`):
           (undocumented)
       * graph_out (:py:obj:`~.ihipGraph`):
           (undocumented)
       * numDependencies_out (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream, hipStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, hipGraph_t * graph_out, const hipGraphNode_t ** dependencies_out, size_t * numDependencies_out)


.. py:function:: hipLaunchHostFunc_spt(stream, fn, userData)

   (No short description, might be part of a group.)

   Args:
       stream (:py:obj:`~.ihipStream_t`/:py:obj:`~.object`):
           (undocumented)

       fn (:py:obj:`~.hipHostFn_t`/:py:obj:`~.object`):
           (undocumented)

       userData (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipLaunchHostFunc_spt(hipStream_t stream, hipHostFn_t fn, void * userData)


.. py:function:: hipGetDriverEntryPoint_spt(symbol, flags)

   (No short description, might be part of a group.)

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       flags (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * funcPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * status (:py:obj:`~.hipDriverEntryPointQueryResult`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetDriverEntryPoint_spt(const char * symbol, void ** funcPtr, unsigned long long flags, hipDriverEntryPointQueryResult * status)


.. py:function:: hipGetProcAddress_spt(symbol, hipVersion, flags)

   (No short description, might be part of a group.)

   Args:
       symbol (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       hipVersion (:py:obj:`~.int`):
           (undocumented)

       flags (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipError_t`: (undocumented)
       * pfn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * symbolStatus (:py:obj:`~.hipDriverProcAddressQueryResult`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipError_t hipGetProcAddress_spt(const char * symbol, void ** pfn, int hipVersion, uint64_t flags, hipDriverProcAddressQueryResult * symbolStatus)


.. py:class:: hipDataType

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIP_R_32F
      :type:  int


   .. py:attribute:: HIP_R_64F
      :type:  int


   .. py:attribute:: HIP_R_16F
      :type:  int


   .. py:attribute:: HIP_R_8I
      :type:  int


   .. py:attribute:: HIP_C_32F
      :type:  int


   .. py:attribute:: HIP_C_64F
      :type:  int


   .. py:attribute:: HIP_C_16F
      :type:  int


   .. py:attribute:: HIP_C_8I
      :type:  int


   .. py:attribute:: HIP_R_8U
      :type:  int


   .. py:attribute:: HIP_C_8U
      :type:  int


   .. py:attribute:: HIP_R_32I
      :type:  int


   .. py:attribute:: HIP_C_32I
      :type:  int


   .. py:attribute:: HIP_R_32U
      :type:  int


   .. py:attribute:: HIP_C_32U
      :type:  int


   .. py:attribute:: HIP_R_16BF
      :type:  int


   .. py:attribute:: HIP_C_16BF
      :type:  int


   .. py:attribute:: HIP_R_4I
      :type:  int


   .. py:attribute:: HIP_C_4I
      :type:  int


   .. py:attribute:: HIP_R_4U
      :type:  int


   .. py:attribute:: HIP_C_4U
      :type:  int


   .. py:attribute:: HIP_R_16I
      :type:  int


   .. py:attribute:: HIP_C_16I
      :type:  int


   .. py:attribute:: HIP_R_16U
      :type:  int


   .. py:attribute:: HIP_C_16U
      :type:  int


   .. py:attribute:: HIP_R_64I
      :type:  int


   .. py:attribute:: HIP_C_64I
      :type:  int


   .. py:attribute:: HIP_R_64U
      :type:  int


   .. py:attribute:: HIP_C_64U
      :type:  int


   .. py:attribute:: HIP_R_8F_E4M3
      :type:  int


   .. py:attribute:: HIP_R_8F_E5M2
      :type:  int


   .. py:attribute:: HIP_R_8F_UE8M0
      :type:  int


   .. py:attribute:: HIP_R_6F_E2M3
      :type:  int


   .. py:attribute:: HIP_R_6F_E3M2
      :type:  int


   .. py:attribute:: HIP_R_4F_E2M1
      :type:  int


   .. py:attribute:: HIP_R_8F_E4M3_FNUZ
      :type:  int


   .. py:attribute:: HIP_R_8F_E5M2_FNUZ
      :type:  int


.. py:class:: hipLibraryPropertyType

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIP_LIBRARY_MAJOR_VERSION
      :type:  int


   .. py:attribute:: HIP_LIBRARY_MINOR_VERSION
      :type:  int


   .. py:attribute:: HIP_LIBRARY_PATCH_LEVEL
      :type:  int


