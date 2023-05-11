# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import ctypes

from hip import hip

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

device_num = 0

for attrib in (
   hip.hipDeviceAttribute_t.hipDeviceAttributeAccessPolicyMaxWindowSize,
   hip.hipDeviceAttribute_t.hipDeviceAttributeAmdSpecificBegin,
   hip.hipDeviceAttribute_t.hipDeviceAttributeAmdSpecificEnd,
   hip.hipDeviceAttribute_t.hipDeviceAttributeArch,
   hip.hipDeviceAttribute_t.hipDeviceAttributeAsicRevision,
   hip.hipDeviceAttribute_t.hipDeviceAttributeAsyncEngineCount,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCanMapHostMemory,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCanUseHostPointerForRegisteredMem,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCanUseStreamWaitValue,
   hip.hipDeviceAttribute_t.hipDeviceAttributeClockRate,
   hip.hipDeviceAttribute_t.hipDeviceAttributeComputeCapabilityMajor,
   hip.hipDeviceAttribute_t.hipDeviceAttributeComputeCapabilityMinor,
   hip.hipDeviceAttribute_t.hipDeviceAttributeComputeMode,
   hip.hipDeviceAttribute_t.hipDeviceAttributeComputePreemptionSupported,
   hip.hipDeviceAttribute_t.hipDeviceAttributeConcurrentKernels,
   hip.hipDeviceAttribute_t.hipDeviceAttributeConcurrentManagedAccess,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCooperativeLaunch,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCooperativeMultiDeviceLaunch,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCudaCompatibleBegin,
   hip.hipDeviceAttribute_t.hipDeviceAttributeCudaCompatibleEnd,
   hip.hipDeviceAttribute_t.hipDeviceAttributeDeviceOverlap,
   hip.hipDeviceAttribute_t.hipDeviceAttributeDirectManagedMemAccessFromHost,
   hip.hipDeviceAttribute_t.hipDeviceAttributeFineGrainSupport,
   hip.hipDeviceAttribute_t.hipDeviceAttributeGcnArch,
   hip.hipDeviceAttribute_t.hipDeviceAttributeGcnArchName,
   hip.hipDeviceAttribute_t.hipDeviceAttributeGlobalL1CacheSupported,
   hip.hipDeviceAttribute_t.hipDeviceAttributeHdpMemFlushCntl,
   hip.hipDeviceAttribute_t.hipDeviceAttributeHdpRegFlushCntl,
   hip.hipDeviceAttribute_t.hipDeviceAttributeHostNativeAtomicSupported,
   hip.hipDeviceAttribute_t.hipDeviceAttributeImageSupport,
   hip.hipDeviceAttribute_t.hipDeviceAttributeIntegrated,
   hip.hipDeviceAttribute_t.hipDeviceAttributeIsLargeBar,
   hip.hipDeviceAttribute_t.hipDeviceAttributeIsMultiGpuBoard,
   hip.hipDeviceAttribute_t.hipDeviceAttributeKernelExecTimeout,
   hip.hipDeviceAttribute_t.hipDeviceAttributeL2CacheSize,
   hip.hipDeviceAttribute_t.hipDeviceAttributeLocalL1CacheSupported,
   hip.hipDeviceAttribute_t.hipDeviceAttributeLuid,
   hip.hipDeviceAttribute_t.hipDeviceAttributeLuidDeviceNodeMask,
   hip.hipDeviceAttribute_t.hipDeviceAttributeManagedMemory,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlockDimX,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlockDimY,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlockDimZ,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlocksPerMultiProcessor,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimX,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimY,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimZ,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxPitch,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxRegistersPerBlock,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxRegistersPerMultiprocessor,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxSharedMemoryPerBlock,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxSurface1D,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxSurface1DLayered,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxSurface2D,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxSurface2DLayered,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxSurface3D,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxSurfaceCubemap,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxSurfaceCubemapLayered,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture1DLayered,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture1DLinear,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture1DMipmap,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture1DWidth,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture2DGather,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture2DHeight,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture2DLayered,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture2DLinear,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture2DMipmap,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture2DWidth,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture3DAlt,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture3DDepth,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture3DHeight,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTexture3DWidth,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTextureCubemap,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxTextureCubemapLayered,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxThreadsDim,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxThreadsPerBlock,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxThreadsPerMultiProcessor,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMemoryBusWidth,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMemoryClockRate,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMemoryPoolsSupported,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMultiGpuBoardGroupID,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMultiprocessorCount,
   hip.hipDeviceAttribute_t.hipDeviceAttributeName,
   hip.hipDeviceAttribute_t.hipDeviceAttributePageableMemoryAccess,
   hip.hipDeviceAttribute_t.hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
   hip.hipDeviceAttribute_t.hipDeviceAttributePciBusId,
   hip.hipDeviceAttribute_t.hipDeviceAttributePciDeviceId,
   hip.hipDeviceAttribute_t.hipDeviceAttributePciDomainID,
   hip.hipDeviceAttribute_t.hipDeviceAttributePersistingL2CacheMaxSize,
   hip.hipDeviceAttribute_t.hipDeviceAttributePhysicalMultiProcessorCount,
   hip.hipDeviceAttribute_t.hipDeviceAttributeReservedSharedMemPerBlock,
   hip.hipDeviceAttribute_t.hipDeviceAttributeSharedMemPerBlockOptin,
   hip.hipDeviceAttribute_t.hipDeviceAttributeSharedMemPerMultiprocessor,
   hip.hipDeviceAttribute_t.hipDeviceAttributeSingleToDoublePrecisionPerfRatio,
   hip.hipDeviceAttribute_t.hipDeviceAttributeStreamPrioritiesSupported,
   hip.hipDeviceAttribute_t.hipDeviceAttributeSurfaceAlignment,
   hip.hipDeviceAttribute_t.hipDeviceAttributeTccDriver,
   hip.hipDeviceAttribute_t.hipDeviceAttributeTextureAlignment,
   hip.hipDeviceAttribute_t.hipDeviceAttributeTexturePitchAlignment,
   hip.hipDeviceAttribute_t.hipDeviceAttributeTotalConstantMemory,
   hip.hipDeviceAttribute_t.hipDeviceAttributeTotalGlobalMem,
   hip.hipDeviceAttribute_t.hipDeviceAttributeUnifiedAddressing,
   hip.hipDeviceAttribute_t.hipDeviceAttributeUuid,
   hip.hipDeviceAttribute_t.hipDeviceAttributeVendorSpecificBegin,
   hip.hipDeviceAttribute_t.hipDeviceAttributeVirtualMemoryManagementSupported,
   hip.hipDeviceAttribute_t.hipDeviceAttributeWallClockRate,
   hip.hipDeviceAttribute_t.hipDeviceAttributeWarpSize,
):
    try:
        if attrib in (
            hip.hipDeviceAttribute_t.hipDeviceAttributeArch,
            hip.hipDeviceAttribute_t.hipDeviceAttributeName,
            hip.hipDeviceAttribute_t.hipDeviceAttributeGcnArch,
            hip.hipDeviceAttribute_t.hipDeviceAttributeGcnArchName,
        ):
            buffer = bytearray(256) # TODO all string attribs fail, need to check if this is expected.
            hip_check(hip.hipDeviceGetAttribute(buffer,attrib,device_num))
            print(f"{attrib.name}: {buffer}")
        else:
            result = ctypes.c_int(device_num)
            hip_check(hip.hipDeviceGetAttribute(ctypes.addressof(result),attrib,device_num))
            print(f"{attrib.name}: {result.value}")
    except:
        print(f"{attrib.name}: hipErrorInvalidValue")
