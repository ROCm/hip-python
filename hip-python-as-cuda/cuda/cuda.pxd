# MIT License
# 
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

cimport hip.chip
cimport hip.hip
from cuda.nvrtc cimport CUlinkState_st
from cuda.nvrtc cimport _nvrtcProgram

cimport cuda.ccuda
cdef class CUuuid_st(hip.hip.hipUUID_t):
    pass
cdef class cudaDeviceProp(hip.hip.hipDeviceProp_t):
    pass
cdef class cudaPointerAttributes(hip.hip.hipPointerAttribute_t):
    pass
cdef class cudaChannelFormatDesc(hip.hip.hipChannelFormatDesc):
    pass
cdef class CUarray_st(hip.hip.hipArray):
    pass
cdef class cudaArray(hip.hip.hipArray):
    pass
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
cdef class CUmipmappedArray_st(hip.hip.hipMipmappedArray):
    pass
cdef class cudaMipmappedArray(hip.hip.hipMipmappedArray):
    pass
cdef class CUDA_TEXTURE_DESC_st(hip.hip.HIP_TEXTURE_DESC_st):
    pass
cdef class cudaResourceDesc(hip.hip.hipResourceDesc):
    pass
cdef class CUDA_RESOURCE_DESC_st(hip.hip.HIP_RESOURCE_DESC_st):
    pass
cdef class cudaResourceViewDesc(hip.hip.hipResourceViewDesc):
    pass
cdef class CUDA_RESOURCE_VIEW_DESC_st(hip.hip.HIP_RESOURCE_VIEW_DESC_st):
    pass
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
cdef class CUmemLocation(hip.hip.hipMemLocation):
    pass
cdef class CUmemLocation_st(hip.hip.hipMemLocation):
    pass
cdef class CUmemLocation_v1(hip.hip.hipMemLocation):
    pass
cdef class cudaMemLocation(hip.hip.hipMemLocation):
    pass
cdef class CUmemcpyAttributes(hip.hip.hipMemcpyAttributes):
    pass
cdef class CUmemcpyAttributes_st(hip.hip.hipMemcpyAttributes):
    pass
cdef class CUmemcpyAttributes_v1(hip.hip.hipMemcpyAttributes):
    pass
cdef class cudaMemcpyAttributes(hip.hip.hipMemcpyAttributes):
    pass
cdef class CUoffset3D(hip.hip.hipOffset3D):
    pass
cdef class CUoffset3D_st(hip.hip.hipOffset3D):
    pass
cdef class CUoffset3D_v1(hip.hip.hipOffset3D):
    pass
cdef class cudaOffset3D(hip.hip.hipOffset3D):
    pass
cdef class CUmemcpy3DOperand(hip.hip.hipMemcpy3DOperand):
    pass
cdef class CUmemcpy3DOperand_st(hip.hip.hipMemcpy3DOperand):
    pass
cdef class CUmemcpy3DOperand_v1(hip.hip.hipMemcpy3DOperand):
    pass
cdef class cudaMemcpy3DOperand(hip.hip.hipMemcpy3DOperand):
    pass
cdef class CUDA_MEMCPY3D_BATCH_OP(hip.hip.hipMemcpy3DBatchOp):
    pass
cdef class CUDA_MEMCPY3D_BATCH_OP_st(hip.hip.hipMemcpy3DBatchOp):
    pass
cdef class CUDA_MEMCPY3D_BATCH_OP_v1(hip.hip.hipMemcpy3DBatchOp):
    pass
cdef class cudaMemcpy3DBatchOp(hip.hip.hipMemcpy3DBatchOp):
    pass
cdef class cudaMemcpy3DPeerParms(hip.hip.hipMemcpy3DPeerParms):
    pass
cdef class uchar1(hip.hip.uchar1):
    pass
cdef class uchar2(hip.hip.uchar2):
    pass
cdef class uchar3(hip.hip.uchar3):
    pass
cdef class uchar4(hip.hip.uchar4):
    pass
cdef class char1(hip.hip.char1):
    pass
cdef class char2(hip.hip.char2):
    pass
cdef class char3(hip.hip.char3):
    pass
cdef class char4(hip.hip.char4):
    pass
cdef class ushort1(hip.hip.ushort1):
    pass
cdef class ushort2(hip.hip.ushort2):
    pass
cdef class ushort3(hip.hip.ushort3):
    pass
cdef class ushort4(hip.hip.ushort4):
    pass
cdef class short1(hip.hip.short1):
    pass
cdef class short2(hip.hip.short2):
    pass
cdef class short3(hip.hip.short3):
    pass
cdef class short4(hip.hip.short4):
    pass
cdef class uint1(hip.hip.uint1):
    pass
cdef class uint2(hip.hip.uint2):
    pass
cdef class uint3(hip.hip.uint3):
    pass
cdef class uint4(hip.hip.uint4):
    pass
cdef class int1(hip.hip.int1):
    pass
cdef class int2(hip.hip.int2):
    pass
cdef class int3(hip.hip.int3):
    pass
cdef class int4(hip.hip.int4):
    pass
cdef class ulong1(hip.hip.ulong1):
    pass
cdef class ulong2(hip.hip.ulong2):
    pass
cdef class ulong3(hip.hip.ulong3):
    pass
cdef class ulong4(hip.hip.ulong4):
    pass
cdef class long1(hip.hip.long1):
    pass
cdef class long2(hip.hip.long2):
    pass
cdef class long3(hip.hip.long3):
    pass
cdef class long4(hip.hip.long4):
    pass
cdef class ulonglong1(hip.hip.ulonglong1):
    pass
cdef class ulonglong2(hip.hip.ulonglong2):
    pass
cdef class ulonglong3(hip.hip.ulonglong3):
    pass
cdef class ulonglong4(hip.hip.ulonglong4):
    pass
cdef class longlong1(hip.hip.longlong1):
    pass
cdef class longlong2(hip.hip.longlong2):
    pass
cdef class longlong3(hip.hip.longlong3):
    pass
cdef class longlong4(hip.hip.longlong4):
    pass
cdef class float1(hip.hip.float1):
    pass
cdef class float2(hip.hip.float2):
    pass
cdef class float3(hip.hip.float3):
    pass
cdef class float4(hip.hip.float4):
    pass
cdef class double1(hip.hip.double1):
    pass
cdef class double2(hip.hip.double2):
    pass
cdef class double3(hip.hip.double3):
    pass
cdef class double4(hip.hip.double4):
    pass
cdef class CUtexref_st(hip.hip.textureReference):
    pass
cdef class textureReference(hip.hip.textureReference):
    pass
cdef class cudaTextureDesc(hip.hip.hipTextureDesc):
    pass
cdef class surfaceReference(hip.hip.surfaceReference):
    pass
cdef class CUctx_st(hip.hip.ihipCtx_t):
    pass
cdef class CUstream_st(hip.hip.ihipStream_t):
    pass
cdef class CUipcMemHandle_st(hip.hip.hipIpcMemHandle_st):
    pass
cdef class cudaIpcMemHandle_st(hip.hip.hipIpcMemHandle_st):
    pass
cdef class CUipcEventHandle_st(hip.hip.hipIpcEventHandle_st):
    pass
cdef class cudaIpcEventHandle_st(hip.hip.hipIpcEventHandle_st):
    pass
cdef class CUmod_st(hip.hip.ihipModule_t):
    pass
cdef class CUfunc_st(hip.hip.ihipModuleSymbol_t):
    pass
cdef class CUlib_st(hip.hip.ihipLibrary_t):
    pass
cdef class CUkern_st(hip.hip.ihipKernel_t):
    pass
cdef class CUmemPoolHandle_st(hip.hip.ihipMemPoolHandle_t):
    pass
cdef class cudaFuncAttributes(hip.hip.hipFuncAttributes):
    pass
cdef class CUevent_st(hip.hip.ihipEvent_t):
    pass
cdef class CUstreamBatchMemOpParams_union(hip.hip.hipStreamBatchMemOpParams_union):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS(hip.hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_st(hip.hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v1(hip.hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st(hip.hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v2(hip.hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st(hip.hip.hipBatchMemOpNodeParams):
    pass
cdef class CUmemAccessDesc(hip.hip.hipMemAccessDesc):
    pass
cdef class CUmemAccessDesc_st(hip.hip.hipMemAccessDesc):
    pass
cdef class CUmemAccessDesc_v1(hip.hip.hipMemAccessDesc):
    pass
cdef class cudaMemAccessDesc(hip.hip.hipMemAccessDesc):
    pass
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
cdef class CUDA_LAUNCH_PARAMS_st(hip.hip.hipFunctionLaunchParams_t):
    pass
cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st(hip.hip.hipExternalMemoryHandleDesc_st):
    pass
cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(hip.hip.hipExternalMemoryBufferDesc_st):
    pass
cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st(hip.hip.hipExternalSemaphoreHandleDesc_st):
    pass
cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(hip.hip.hipExternalSemaphoreSignalParams_st):
    pass
cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(hip.hip.hipExternalSemaphoreWaitParams_st):
    pass
cdef class CUgraph_st(hip.hip.ihipGraph):
    pass
cdef class CUgraphNode_st(hip.hip.hipGraphNode):
    pass
cdef class CUgraphExec_st(hip.hip.hipGraphExec):
    pass
cdef class CUuserObject_st(hip.hip.hipUserObject):
    pass
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
cdef class CUDA_MEM_ALLOC_NODE_PARAMS(hip.hip.hipMemAllocNodeParams):
    pass
cdef class CUDA_MEM_ALLOC_NODE_PARAMS_st(hip.hip.hipMemAllocNodeParams):
    pass
cdef class CUDA_MEM_ALLOC_NODE_PARAMS_v1(hip.hip.hipMemAllocNodeParams):
    pass
cdef class CUDA_MEM_ALLOC_NODE_PARAMS_v1_st(hip.hip.hipMemAllocNodeParams):
    pass
cdef class cudaMemAllocNodeParams(hip.hip.hipMemAllocNodeParams):
    pass
cdef class CUaccessPolicyWindow(hip.hip.hipAccessPolicyWindow):
    pass
cdef class CUaccessPolicyWindow_st(hip.hip.hipAccessPolicyWindow):
    pass
cdef class cudaAccessPolicyWindow(hip.hip.hipAccessPolicyWindow):
    pass
cdef class CUlaunchMemSyncDomainMap(hip.hip.hipLaunchMemSyncDomainMap):
    pass
cdef class CUlaunchMemSyncDomainMap_st(hip.hip.hipLaunchMemSyncDomainMap):
    pass
cdef class cudaLaunchMemSyncDomainMap(hip.hip.hipLaunchMemSyncDomainMap):
    pass
cdef class cudaLaunchMemSyncDomainMap_st(hip.hip.hipLaunchMemSyncDomainMap):
    pass
cdef class CUlaunchAttributeValue(hip.hip.hipLaunchAttributeValue):
    pass
cdef class CUlaunchAttributeValue_union(hip.hip.hipLaunchAttributeValue):
    pass
cdef class CUstreamAttrValue(hip.hip.hipLaunchAttributeValue):
    pass
cdef class CUstreamAttrValue_union(hip.hip.hipLaunchAttributeValue):
    pass
cdef class CUstreamAttrValue_v1(hip.hip.hipLaunchAttributeValue):
    pass
cdef class cudaLaunchAttributeValue(hip.hip.hipLaunchAttributeValue):
    pass
cdef class cudaStreamAttrValue(hip.hip.hipLaunchAttributeValue):
    pass
cdef class CUDA_GRAPH_INSTANTIATE_PARAMS(hip.hip.hipGraphInstantiateParams):
    pass
cdef class CUDA_GRAPH_INSTANTIATE_PARAMS_st(hip.hip.hipGraphInstantiateParams):
    pass
cdef class cudaGraphInstantiateParams(hip.hip.hipGraphInstantiateParams):
    pass
cdef class cudaGraphInstantiateParams_st(hip.hip.hipGraphInstantiateParams):
    pass
cdef class CUmemAllocationProp(hip.hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_st(hip.hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_v1(hip.hip.hipMemAllocationProp):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS(hip.hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st(hip.hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1(hip.hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2(hip.hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st(hip.hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class cudaExternalSemaphoreSignalNodeParams(hip.hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class cudaExternalSemaphoreSignalNodeParamsV2(hip.hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS(hip.hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_st(hip.hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1(hip.hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2(hip.hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st(hip.hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class cudaExternalSemaphoreWaitNodeParams(hip.hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class cudaExternalSemaphoreWaitNodeParamsV2(hip.hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUarrayMapInfo(hip.hip.hipArrayMapInfo):
    pass
cdef class CUarrayMapInfo_st(hip.hip.hipArrayMapInfo):
    pass
cdef class CUarrayMapInfo_v1(hip.hip.hipArrayMapInfo):
    pass
cdef class CUDA_MEMCPY_NODE_PARAMS(hip.hip.hipMemcpyNodeParams):
    pass
cdef class CUDA_MEMCPY_NODE_PARAMS_st(hip.hip.hipMemcpyNodeParams):
    pass
cdef class cudaMemcpyNodeParams(hip.hip.hipMemcpyNodeParams):
    pass
cdef class CUDA_CHILD_GRAPH_NODE_PARAMS(hip.hip.hipChildGraphNodeParams):
    pass
cdef class CUDA_CHILD_GRAPH_NODE_PARAMS_st(hip.hip.hipChildGraphNodeParams):
    pass
cdef class cudaChildGraphNodeParams(hip.hip.hipChildGraphNodeParams):
    pass
cdef class CUDA_EVENT_WAIT_NODE_PARAMS(hip.hip.hipEventWaitNodeParams):
    pass
cdef class CUDA_EVENT_WAIT_NODE_PARAMS_st(hip.hip.hipEventWaitNodeParams):
    pass
cdef class cudaEventWaitNodeParams(hip.hip.hipEventWaitNodeParams):
    pass
cdef class CUDA_EVENT_RECORD_NODE_PARAMS(hip.hip.hipEventRecordNodeParams):
    pass
cdef class CUDA_EVENT_RECORD_NODE_PARAMS_st(hip.hip.hipEventRecordNodeParams):
    pass
cdef class cudaEventRecordNodeParams(hip.hip.hipEventRecordNodeParams):
    pass
cdef class CUDA_MEM_FREE_NODE_PARAMS(hip.hip.hipMemFreeNodeParams):
    pass
cdef class CUDA_MEM_FREE_NODE_PARAMS_st(hip.hip.hipMemFreeNodeParams):
    pass
cdef class cudaMemFreeNodeParams(hip.hip.hipMemFreeNodeParams):
    pass
cdef class CUgraphNodeParams(hip.hip.hipGraphNodeParams):
    pass
cdef class CUgraphNodeParams_st(hip.hip.hipGraphNodeParams):
    pass
cdef class cudaGraphNodeParams(hip.hip.hipGraphNodeParams):
    pass
cdef class CUgraphEdgeData(hip.hip.hipGraphEdgeData):
    pass
cdef class CUgraphEdgeData_st(hip.hip.hipGraphEdgeData):
    pass
cdef class cudaGraphEdgeData(hip.hip.hipGraphEdgeData):
    pass
cdef class cudaGraphEdgeData_st(hip.hip.hipGraphEdgeData):
    pass
cdef class CUlaunchAttribute_st(hip.hip.hipLaunchAttribute_st):
    pass
cdef class cudaLaunchAttribute_st(hip.hip.hipLaunchAttribute_st):
    pass
cdef class cudaLaunchConfig_st(hip.hip.hipLaunchConfig_st):
    pass
cdef class CUlaunchConfig_st(hip.hip.HIP_LAUNCH_CONFIG_st):
    pass
cdef class CUstreamCallback(hip.hip.hipStreamCallback_t):
    pass
cdef class cudaStreamCallback_t(hip.hip.hipStreamCallback_t):
    pass