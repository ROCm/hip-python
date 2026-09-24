# MIT License
# 
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

cimport rocm.bindings.cyhip as cyhip
cimport rocm.bindings.hip as hip
from cuda.bindings.nvrtc cimport CUlinkState_st
from cuda.bindings.nvrtc cimport _nvrtcProgram

cimport cuda.bindings.cydriver
cdef class CUuuid_st(hip.hipUUID_t):
    pass
cdef class cudaDeviceProp(hip.hipDeviceProp_t):
    pass
cdef class cudaPointerAttributes(hip.hipPointerAttribute_t):
    pass
cdef class cudaChannelFormatDesc(hip.hipChannelFormatDesc):
    pass
cdef class CUarray_st(hip.hipArray):
    pass
cdef class cudaArray(hip.hipArray):
    pass
cdef class CUDA_ARRAY_DESCRIPTOR(hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY_DESCRIPTOR_st(hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY_DESCRIPTOR_v1(hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY_DESCRIPTOR_v1_st(hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY_DESCRIPTOR_v2(hip.HIP_ARRAY_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY3D_DESCRIPTOR(hip.HIP_ARRAY3D_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY3D_DESCRIPTOR_st(hip.HIP_ARRAY3D_DESCRIPTOR):
    pass
cdef class CUDA_ARRAY3D_DESCRIPTOR_v2(hip.HIP_ARRAY3D_DESCRIPTOR):
    pass
cdef class CUDA_MEMCPY2D(hip.hip_Memcpy2D):
    pass
cdef class CUDA_MEMCPY2D_st(hip.hip_Memcpy2D):
    pass
cdef class CUDA_MEMCPY2D_v1(hip.hip_Memcpy2D):
    pass
cdef class CUDA_MEMCPY2D_v1_st(hip.hip_Memcpy2D):
    pass
cdef class CUDA_MEMCPY2D_v2(hip.hip_Memcpy2D):
    pass
cdef class CUmipmappedArray_st(hip.hipMipmappedArray):
    pass
cdef class cudaMipmappedArray(hip.hipMipmappedArray):
    pass
cdef class CUDA_TEXTURE_DESC_st(hip.HIP_TEXTURE_DESC_st):
    pass
cdef class cudaResourceDesc(hip.hipResourceDesc):
    pass
cdef class CUDA_RESOURCE_DESC_st(hip.HIP_RESOURCE_DESC_st):
    pass
cdef class cudaResourceViewDesc(hip.hipResourceViewDesc):
    pass
cdef class CUDA_RESOURCE_VIEW_DESC_st(hip.HIP_RESOURCE_VIEW_DESC_st):
    pass
cdef class cudaPitchedPtr(hip.hipPitchedPtr):
    pass
cdef class cudaExtent(hip.hipExtent):
    pass
cdef class cudaPos(hip.hipPos):
    pass
cdef class cudaMemcpy3DParms(hip.hipMemcpy3DParms):
    pass
cdef class CUDA_MEMCPY3D(hip.HIP_MEMCPY3D):
    pass
cdef class CUDA_MEMCPY3D_st(hip.HIP_MEMCPY3D):
    pass
cdef class CUDA_MEMCPY3D_v1(hip.HIP_MEMCPY3D):
    pass
cdef class CUDA_MEMCPY3D_v1_st(hip.HIP_MEMCPY3D):
    pass
cdef class CUDA_MEMCPY3D_v2(hip.HIP_MEMCPY3D):
    pass
cdef class CUmemLocation(hip.hipMemLocation):
    pass
cdef class CUmemLocation_st(hip.hipMemLocation):
    pass
cdef class CUmemLocation_v1(hip.hipMemLocation):
    pass
cdef class cudaMemLocation(hip.hipMemLocation):
    pass
cdef class CUmemcpyAttributes(hip.hipMemcpyAttributes):
    pass
cdef class CUmemcpyAttributes_st(hip.hipMemcpyAttributes):
    pass
cdef class CUmemcpyAttributes_v1(hip.hipMemcpyAttributes):
    pass
cdef class cudaMemcpyAttributes(hip.hipMemcpyAttributes):
    pass
cdef class CUoffset3D(hip.hipOffset3D):
    pass
cdef class CUoffset3D_st(hip.hipOffset3D):
    pass
cdef class CUoffset3D_v1(hip.hipOffset3D):
    pass
cdef class cudaOffset3D(hip.hipOffset3D):
    pass
cdef class CUmemcpy3DOperand(hip.hipMemcpy3DOperand):
    pass
cdef class CUmemcpy3DOperand_st(hip.hipMemcpy3DOperand):
    pass
cdef class CUmemcpy3DOperand_v1(hip.hipMemcpy3DOperand):
    pass
cdef class cudaMemcpy3DOperand(hip.hipMemcpy3DOperand):
    pass
cdef class CUDA_MEMCPY3D_BATCH_OP(hip.hipMemcpy3DBatchOp):
    pass
cdef class CUDA_MEMCPY3D_BATCH_OP_st(hip.hipMemcpy3DBatchOp):
    pass
cdef class CUDA_MEMCPY3D_BATCH_OP_v1(hip.hipMemcpy3DBatchOp):
    pass
cdef class cudaMemcpy3DBatchOp(hip.hipMemcpy3DBatchOp):
    pass
cdef class cudaMemcpy3DPeerParms(hip.hipMemcpy3DPeerParms):
    pass
cdef class uchar1(hip.uchar1):
    pass
cdef class uchar2(hip.uchar2):
    pass
cdef class uchar3(hip.uchar3):
    pass
cdef class uchar4(hip.uchar4):
    pass
cdef class char1(hip.char1):
    pass
cdef class char2(hip.char2):
    pass
cdef class char3(hip.char3):
    pass
cdef class char4(hip.char4):
    pass
cdef class ushort1(hip.ushort1):
    pass
cdef class ushort2(hip.ushort2):
    pass
cdef class ushort3(hip.ushort3):
    pass
cdef class ushort4(hip.ushort4):
    pass
cdef class short1(hip.short1):
    pass
cdef class short2(hip.short2):
    pass
cdef class short3(hip.short3):
    pass
cdef class short4(hip.short4):
    pass
cdef class uint1(hip.uint1):
    pass
cdef class uint2(hip.uint2):
    pass
cdef class uint3(hip.uint3):
    pass
cdef class uint4(hip.uint4):
    pass
cdef class int1(hip.int1):
    pass
cdef class int2(hip.int2):
    pass
cdef class int3(hip.int3):
    pass
cdef class int4(hip.int4):
    pass
cdef class ulong1(hip.ulong1):
    pass
cdef class ulong2(hip.ulong2):
    pass
cdef class ulong3(hip.ulong3):
    pass
cdef class ulong4(hip.ulong4):
    pass
cdef class long1(hip.long1):
    pass
cdef class long2(hip.long2):
    pass
cdef class long3(hip.long3):
    pass
cdef class long4(hip.long4):
    pass
cdef class ulonglong1(hip.ulonglong1):
    pass
cdef class ulonglong2(hip.ulonglong2):
    pass
cdef class ulonglong3(hip.ulonglong3):
    pass
cdef class ulonglong4(hip.ulonglong4):
    pass
cdef class longlong1(hip.longlong1):
    pass
cdef class longlong2(hip.longlong2):
    pass
cdef class longlong3(hip.longlong3):
    pass
cdef class longlong4(hip.longlong4):
    pass
cdef class float1(hip.float1):
    pass
cdef class float2(hip.float2):
    pass
cdef class float3(hip.float3):
    pass
cdef class float4(hip.float4):
    pass
cdef class double1(hip.double1):
    pass
cdef class double2(hip.double2):
    pass
cdef class double3(hip.double3):
    pass
cdef class double4(hip.double4):
    pass
cdef class CUtexref_st(hip.textureReference):
    pass
cdef class textureReference(hip.textureReference):
    pass
cdef class cudaTextureDesc(hip.hipTextureDesc):
    pass
cdef class surfaceReference(hip.surfaceReference):
    pass
cdef class CUctx_st(hip.ihipCtx_t):
    pass
cdef class CUstream_st(hip.ihipStream_t):
    pass
cdef class CUipcMemHandle_st(hip.hipIpcMemHandle_st):
    pass
cdef class cudaIpcMemHandle_st(hip.hipIpcMemHandle_st):
    pass
cdef class CUipcEventHandle_st(hip.hipIpcEventHandle_st):
    pass
cdef class cudaIpcEventHandle_st(hip.hipIpcEventHandle_st):
    pass
cdef class CUmod_st(hip.ihipModule_t):
    pass
cdef class CUfunc_st(hip.ihipModuleSymbol_t):
    pass
cdef class CUlib_st(hip.ihipLibrary_t):
    pass
cdef class CUkern_st(hip.ihipKernel_t):
    pass
cdef class CUmemPoolHandle_st(hip.ihipMemPoolHandle_t):
    pass
cdef class cudaFuncAttributes(hip.hipFuncAttributes):
    pass
cdef class CUevent_st(hip.ihipEvent_t):
    pass
cdef class CUstreamBatchMemOpParams_union(hip.hipStreamBatchMemOpParams_union):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS(hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_st(hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v1(hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st(hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v2(hip.hipBatchMemOpNodeParams):
    pass
cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st(hip.hipBatchMemOpNodeParams):
    pass
cdef class CUmemAccessDesc(hip.hipMemAccessDesc):
    pass
cdef class CUmemAccessDesc_st(hip.hipMemAccessDesc):
    pass
cdef class CUmemAccessDesc_v1(hip.hipMemAccessDesc):
    pass
cdef class cudaMemAccessDesc(hip.hipMemAccessDesc):
    pass
cdef class CUmemPoolProps(hip.hipMemPoolProps):
    pass
cdef class CUmemPoolProps_st(hip.hipMemPoolProps):
    pass
cdef class CUmemPoolProps_v1(hip.hipMemPoolProps):
    pass
cdef class cudaMemPoolProps(hip.hipMemPoolProps):
    pass
cdef class CUmemPoolPtrExportData(hip.hipMemPoolPtrExportData):
    pass
cdef class CUmemPoolPtrExportData_st(hip.hipMemPoolPtrExportData):
    pass
cdef class CUmemPoolPtrExportData_v1(hip.hipMemPoolPtrExportData):
    pass
cdef class cudaMemPoolPtrExportData(hip.hipMemPoolPtrExportData):
    pass
cdef class CUDA_LAUNCH_PARAMS_st(hip.hipFunctionLaunchParams_t):
    pass
cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st(hip.hipExternalMemoryHandleDesc_st):
    pass
cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(hip.hipExternalMemoryBufferDesc_st):
    pass
cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st(hip.hipExternalSemaphoreHandleDesc_st):
    pass
cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(hip.hipExternalSemaphoreSignalParams_st):
    pass
cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(hip.hipExternalSemaphoreWaitParams_st):
    pass
cdef class CUgraph_st(hip.ihipGraph):
    pass
cdef class CUgraphNode_st(hip.hipGraphNode):
    pass
cdef class CUgraphExec_st(hip.hipGraphExec):
    pass
cdef class CUuserObject_st(hip.hipUserObject):
    pass
cdef class CUhostFn(hip.hipHostFn_t):
    pass
cdef class cudaHostFn_t(hip.hipHostFn_t):
    pass
cdef class CUDA_HOST_NODE_PARAMS(hip.hipHostNodeParams):
    pass
cdef class CUDA_HOST_NODE_PARAMS_st(hip.hipHostNodeParams):
    pass
cdef class CUDA_HOST_NODE_PARAMS_v1(hip.hipHostNodeParams):
    pass
cdef class cudaHostNodeParams(hip.hipHostNodeParams):
    pass
cdef class CUDA_KERNEL_NODE_PARAMS(hip.hipKernelNodeParams):
    pass
cdef class CUDA_KERNEL_NODE_PARAMS_st(hip.hipKernelNodeParams):
    pass
cdef class CUDA_KERNEL_NODE_PARAMS_v1(hip.hipKernelNodeParams):
    pass
cdef class cudaKernelNodeParams(hip.hipKernelNodeParams):
    pass
cdef class CUDA_MEMSET_NODE_PARAMS(hip.hipMemsetParams):
    pass
cdef class CUDA_MEMSET_NODE_PARAMS_st(hip.hipMemsetParams):
    pass
cdef class CUDA_MEMSET_NODE_PARAMS_v1(hip.hipMemsetParams):
    pass
cdef class cudaMemsetParams(hip.hipMemsetParams):
    pass
cdef class CUDA_MEM_ALLOC_NODE_PARAMS(hip.hipMemAllocNodeParams):
    pass
cdef class CUDA_MEM_ALLOC_NODE_PARAMS_st(hip.hipMemAllocNodeParams):
    pass
cdef class CUDA_MEM_ALLOC_NODE_PARAMS_v1(hip.hipMemAllocNodeParams):
    pass
cdef class CUDA_MEM_ALLOC_NODE_PARAMS_v1_st(hip.hipMemAllocNodeParams):
    pass
cdef class cudaMemAllocNodeParams(hip.hipMemAllocNodeParams):
    pass
cdef class CUaccessPolicyWindow(hip.hipAccessPolicyWindow):
    pass
cdef class CUaccessPolicyWindow_st(hip.hipAccessPolicyWindow):
    pass
cdef class cudaAccessPolicyWindow(hip.hipAccessPolicyWindow):
    pass
cdef class CUlaunchMemSyncDomainMap(hip.hipLaunchMemSyncDomainMap):
    pass
cdef class CUlaunchMemSyncDomainMap_st(hip.hipLaunchMemSyncDomainMap):
    pass
cdef class cudaLaunchMemSyncDomainMap(hip.hipLaunchMemSyncDomainMap):
    pass
cdef class cudaLaunchMemSyncDomainMap_st(hip.hipLaunchMemSyncDomainMap):
    pass
cdef class CUlaunchAttributeValue(hip.hipLaunchAttributeValue):
    pass
cdef class CUlaunchAttributeValue_union(hip.hipLaunchAttributeValue):
    pass
cdef class CUstreamAttrValue(hip.hipLaunchAttributeValue):
    pass
cdef class CUstreamAttrValue_union(hip.hipLaunchAttributeValue):
    pass
cdef class CUstreamAttrValue_v1(hip.hipLaunchAttributeValue):
    pass
cdef class cudaLaunchAttributeValue(hip.hipLaunchAttributeValue):
    pass
cdef class cudaStreamAttrValue(hip.hipLaunchAttributeValue):
    pass
cdef class CUDA_GRAPH_INSTANTIATE_PARAMS(hip.hipGraphInstantiateParams):
    pass
cdef class CUDA_GRAPH_INSTANTIATE_PARAMS_st(hip.hipGraphInstantiateParams):
    pass
cdef class cudaGraphInstantiateParams(hip.hipGraphInstantiateParams):
    pass
cdef class cudaGraphInstantiateParams_st(hip.hipGraphInstantiateParams):
    pass
cdef class CUmemAllocationProp(hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_st(hip.hipMemAllocationProp):
    pass
cdef class CUmemAllocationProp_v1(hip.hipMemAllocationProp):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS(hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st(hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1(hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2(hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st(hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class cudaExternalSemaphoreSignalNodeParams(hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class cudaExternalSemaphoreSignalNodeParamsV2(hip.hipExternalSemaphoreSignalNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS(hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_st(hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1(hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2(hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st(hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class cudaExternalSemaphoreWaitNodeParams(hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class cudaExternalSemaphoreWaitNodeParamsV2(hip.hipExternalSemaphoreWaitNodeParams):
    pass
cdef class CUarrayMapInfo(hip.hipArrayMapInfo):
    pass
cdef class CUarrayMapInfo_st(hip.hipArrayMapInfo):
    pass
cdef class CUarrayMapInfo_v1(hip.hipArrayMapInfo):
    pass
cdef class CUDA_MEMCPY_NODE_PARAMS(hip.hipMemcpyNodeParams):
    pass
cdef class CUDA_MEMCPY_NODE_PARAMS_st(hip.hipMemcpyNodeParams):
    pass
cdef class cudaMemcpyNodeParams(hip.hipMemcpyNodeParams):
    pass
cdef class CUDA_CHILD_GRAPH_NODE_PARAMS(hip.hipChildGraphNodeParams):
    pass
cdef class CUDA_CHILD_GRAPH_NODE_PARAMS_st(hip.hipChildGraphNodeParams):
    pass
cdef class cudaChildGraphNodeParams(hip.hipChildGraphNodeParams):
    pass
cdef class CUDA_EVENT_WAIT_NODE_PARAMS(hip.hipEventWaitNodeParams):
    pass
cdef class CUDA_EVENT_WAIT_NODE_PARAMS_st(hip.hipEventWaitNodeParams):
    pass
cdef class cudaEventWaitNodeParams(hip.hipEventWaitNodeParams):
    pass
cdef class CUDA_EVENT_RECORD_NODE_PARAMS(hip.hipEventRecordNodeParams):
    pass
cdef class CUDA_EVENT_RECORD_NODE_PARAMS_st(hip.hipEventRecordNodeParams):
    pass
cdef class cudaEventRecordNodeParams(hip.hipEventRecordNodeParams):
    pass
cdef class CUDA_MEM_FREE_NODE_PARAMS(hip.hipMemFreeNodeParams):
    pass
cdef class CUDA_MEM_FREE_NODE_PARAMS_st(hip.hipMemFreeNodeParams):
    pass
cdef class cudaMemFreeNodeParams(hip.hipMemFreeNodeParams):
    pass
cdef class CUgraphNodeParams(hip.hipGraphNodeParams):
    pass
cdef class CUgraphNodeParams_st(hip.hipGraphNodeParams):
    pass
cdef class cudaGraphNodeParams(hip.hipGraphNodeParams):
    pass
cdef class CUgraphEdgeData(hip.hipGraphEdgeData):
    pass
cdef class CUgraphEdgeData_st(hip.hipGraphEdgeData):
    pass
cdef class cudaGraphEdgeData(hip.hipGraphEdgeData):
    pass
cdef class cudaGraphEdgeData_st(hip.hipGraphEdgeData):
    pass
cdef class CUlaunchAttribute_st(hip.hipLaunchAttribute_st):
    pass
cdef class cudaLaunchAttribute_st(hip.hipLaunchAttribute_st):
    pass
cdef class cudaLaunchConfig_st(hip.hipLaunchConfig_st):
    pass
cdef class CUlaunchConfig_st(hip.HIP_LAUNCH_CONFIG_st):
    pass
cdef class CUstreamCallback(hip.hipStreamCallback_t):
    pass
cdef class cudaStreamCallback_t(hip.hipStreamCallback_t):
    pass