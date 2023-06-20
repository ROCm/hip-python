# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport hip.chip
cimport hip.hip

cimport cuda.ccudart
from hip.hip cimport hipUUID_t
cdef class CUuuid_st(hip.hip.hipUUID_t):
    pass
from hip.hip cimport hipDeviceProp_t
cdef class cudaDeviceProp(hip.hip.hipDeviceProp_t):
    pass
from hip.hip cimport hipPointerAttribute_t
cdef class cudaPointerAttributes(hip.hip.hipPointerAttribute_t):
    pass
from hip.hip cimport hipChannelFormatDesc
cdef class cudaChannelFormatDesc(hip.hip.hipChannelFormatDesc):
    pass
from hip.hip cimport HIP_ARRAY_DESCRIPTOR
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
from hip.hip cimport HIP_ARRAY3D_DESCRIPTOR
cdef class CUDA_ARRAY3D_DESCRIPTOR(hip.hip.HIP_ARRAY3D_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY3D_DESCRIPTOR_st(hip.hip.HIP_ARRAY3D_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY3D_DESCRIPTOR_v2(hip.hip.HIP_ARRAY3D_DESCRIPTOR):
    pass
from hip.hip cimport hipArray
cdef class CUarray_st(hip.hip.hipArray):
    pass
cdef class cudaArray(hip.hip.hipArray):
    pass
from hip.hip cimport hip_Memcpy2D
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
from hip.hip cimport hipMipmappedArray
cdef class CUmipmappedArray_st(hip.hip.hipMipmappedArray):
    pass
cdef class cudaMipmappedArray(hip.hip.hipMipmappedArray):
    pass
from hip.hip cimport HIP_TEXTURE_DESC_st
cdef class CUDA_TEXTURE_DESC_st(hip.hip.HIP_TEXTURE_DESC_st):
    pass
from hip.hip cimport hipResourceDesc
cdef class cudaResourceDesc(hip.hip.hipResourceDesc):
    pass
from hip.hip cimport HIP_RESOURCE_DESC_st
cdef class CUDA_RESOURCE_DESC_st(hip.hip.HIP_RESOURCE_DESC_st):
    pass
from hip.hip cimport hipResourceViewDesc
cdef class cudaResourceViewDesc(hip.hip.hipResourceViewDesc):
    pass
from hip.hip cimport HIP_RESOURCE_VIEW_DESC_st
cdef class CUDA_RESOURCE_VIEW_DESC_st(hip.hip.HIP_RESOURCE_VIEW_DESC_st):
    pass
from hip.hip cimport hipPitchedPtr
cdef class cudaPitchedPtr(hip.hip.hipPitchedPtr):
    pass
from hip.hip cimport hipExtent
cdef class cudaExtent(hip.hip.hipExtent):
    pass
from hip.hip cimport hipPos
cdef class cudaPos(hip.hip.hipPos):
    pass
from hip.hip cimport hipMemcpy3DParms
cdef class cudaMemcpy3DParms(hip.hip.hipMemcpy3DParms):
    pass
from hip.hip cimport HIP_MEMCPY3D
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
cdef class CUtexref_st(hip.hip.textureReference):
    pass
cdef class textureReference(hip.hip.textureReference):
    pass
from hip.hip cimport hipTextureDesc
cdef class cudaTextureDesc(hip.hip.hipTextureDesc):
    pass
cdef class surfaceReference(hip.hip.surfaceReference):
    pass
from hip.hip cimport ihipCtx_t
cdef class CUctx_st(hip.hip.ihipCtx_t):
    pass
from hip.hip cimport ihipStream_t
cdef class CUstream_st(hip.hip.ihipStream_t):
    pass
from hip.hip cimport hipIpcMemHandle_st
cdef class CUipcMemHandle_st(hip.hip.hipIpcMemHandle_st):
    pass
cdef class cudaIpcMemHandle_st(hip.hip.hipIpcMemHandle_st):
    pass
from hip.hip cimport hipIpcEventHandle_st
cdef class CUipcEventHandle_st(hip.hip.hipIpcEventHandle_st):
    pass
cdef class cudaIpcEventHandle_st(hip.hip.hipIpcEventHandle_st):
    pass
from hip.hip cimport ihipModule_t
cdef class CUmod_st(hip.hip.ihipModule_t):
    pass
from hip.hip cimport ihipModuleSymbol_t
cdef class CUfunc_st(hip.hip.ihipModuleSymbol_t):
    pass
from hip.hip cimport ihipMemPoolHandle_t
cdef class CUmemPoolHandle_st(hip.hip.ihipMemPoolHandle_t):
    pass
from hip.hip cimport hipFuncAttributes
cdef class cudaFuncAttributes(hip.hip.hipFuncAttributes):
    pass
from hip.hip cimport ihipEvent_t
cdef class CUevent_st(hip.hip.ihipEvent_t):
    pass
from hip.hip cimport hipMemLocation
cdef class CUmemLocation(hip.hip.hipMemLocation):
    pass
cdef class CUmemLocation_st(hip.hip.hipMemLocation):
    pass
cdef class CUmemLocation_v1(hip.hip.hipMemLocation):
    pass
cdef class cudaMemLocation(hip.hip.hipMemLocation):
    pass
from hip.hip cimport hipMemAccessDesc
cdef class CUmemAccessDesc(hip.hip.hipMemAccessDesc):
    pass
cdef class CUmemAccessDesc_st(hip.hip.hipMemAccessDesc):
    pass
cdef class CUmemAccessDesc_v1(hip.hip.hipMemAccessDesc):
    pass
cdef class cudaMemAccessDesc(hip.hip.hipMemAccessDesc):
    pass
from hip.hip cimport hipMemPoolProps
cdef class CUmemPoolProps(hip.hip.hipMemPoolProps):
    pass
cdef class CUmemPoolProps_st(hip.hip.hipMemPoolProps):
    pass
cdef class CUmemPoolProps_v1(hip.hip.hipMemPoolProps):
    pass
cdef class cudaMemPoolProps(hip.hip.hipMemPoolProps):
    pass
from hip.hip cimport hipMemPoolPtrExportData
cdef class CUmemPoolPtrExportData(hip.hip.hipMemPoolPtrExportData):
    pass
cdef class CUmemPoolPtrExportData_st(hip.hip.hipMemPoolPtrExportData):
    pass
cdef class CUmemPoolPtrExportData_v1(hip.hip.hipMemPoolPtrExportData):
    pass
cdef class cudaMemPoolPtrExportData(hip.hip.hipMemPoolPtrExportData):
    pass
from hip.hip cimport hipExternalMemoryHandleDesc_st
cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st(hip.hip.hipExternalMemoryHandleDesc_st):
    pass
from hip.hip cimport hipExternalMemoryBufferDesc_st
cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(hip.hip.hipExternalMemoryBufferDesc_st):
    pass
from hip.hip cimport hipExternalSemaphoreHandleDesc_st
cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st(hip.hip.hipExternalSemaphoreHandleDesc_st):
    pass
from hip.hip cimport hipExternalSemaphoreSignalParams_st
cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(hip.hip.hipExternalSemaphoreSignalParams_st):
    pass
from hip.hip cimport hipExternalSemaphoreWaitParams_st
cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(hip.hip.hipExternalSemaphoreWaitParams_st):
    pass
from hip.hip cimport ihipGraph
cdef class CUgraph_st(hip.hip.ihipGraph):
    pass
from hip.hip cimport hipGraphNode
cdef class CUgraphNode_st(hip.hip.hipGraphNode):
    pass
from hip.hip cimport hipGraphExec
cdef class CUgraphExec_st(hip.hip.hipGraphExec):
    pass
from hip.hip cimport hipUserObject
cdef class CUuserObject_st(hip.hip.hipUserObject):
    pass
from hip.hip cimport hipHostFn_t
cdef class CUhostFn(hip.hip.hipHostFn_t):
    pass
cdef class cudaHostFn_t(hip.hip.hipHostFn_t):
    pass
from hip.hip cimport hipHostNodeParams
cdef class CUDA_HOST_NODE_PARAMS(hip.hip.hipHostNodeParams):
    pass
cdef class CUDA_HOST_NODE_PARAMS_st(hip.hip.hipHostNodeParams):
    pass
cdef class CUDA_HOST_NODE_PARAMS_v1(hip.hip.hipHostNodeParams):
    pass
cdef class cudaHostNodeParams(hip.hip.hipHostNodeParams):
    pass
from hip.hip cimport hipKernelNodeParams
cdef class CUDA_KERNEL_NODE_PARAMS(hip.hip.hipKernelNodeParams):
    pass
cdef class CUDA_KERNEL_NODE_PARAMS_st(hip.hip.hipKernelNodeParams):
    pass
cdef class CUDA_KERNEL_NODE_PARAMS_v1(hip.hip.hipKernelNodeParams):
    pass
cdef class cudaKernelNodeParams(hip.hip.hipKernelNodeParams):
    pass
from hip.hip cimport hipMemsetParams
cdef class CUDA_MEMSET_NODE_PARAMS(hip.hip.hipMemsetParams):
    pass
cdef class CUDA_MEMSET_NODE_PARAMS_st(hip.hip.hipMemsetParams):
    pass
cdef class CUDA_MEMSET_NODE_PARAMS_v1(hip.hip.hipMemsetParams):
    pass
cdef class cudaMemsetParams(hip.hip.hipMemsetParams):
    pass
from hip.hip cimport hipAccessPolicyWindow
cdef class CUaccessPolicyWindow(hip.hip.hipAccessPolicyWindow):
    pass
cdef class CUaccessPolicyWindow_st(hip.hip.hipAccessPolicyWindow):
    pass
cdef class cudaAccessPolicyWindow(hip.hip.hipAccessPolicyWindow):
    pass
from hip.hip cimport hipKernelNodeAttrValue
cdef class CUkernelNodeAttrValue(hip.hip.hipKernelNodeAttrValue):
    pass
cdef class CUkernelNodeAttrValue_union(hip.hip.hipKernelNodeAttrValue):
    pass
cdef class CUkernelNodeAttrValue_v1(hip.hip.hipKernelNodeAttrValue):
    pass
cdef class cudaKernelNodeAttrValue(hip.hip.hipKernelNodeAttrValue):
    pass
from hip.hip cimport hipMemAllocationProp
cdef class CUmemAllocationProp(hip.hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_st(hip.hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_v1(hip.hip.hipMemAllocationProp):
    pass
from hip.hip cimport hipArrayMapInfo
cdef class CUarrayMapInfo(hip.hip.hipArrayMapInfo):
    pass
cdef class CUarrayMapInfo_st(hip.hip.hipArrayMapInfo):
    pass
cdef class CUarrayMapInfo_v1(hip.hip.hipArrayMapInfo):
    pass
from hip.hip cimport hipStreamCallback_t
cdef class CUstreamCallback(hip.hip.hipStreamCallback_t):
    pass
cdef class cudaStreamCallback_t(hip.hip.hipStreamCallback_t):
    pass