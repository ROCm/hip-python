# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

from hip import hip

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props,0))

print(f"{props.asicRevision=}")
print(f"{props.canMapHostMemory=}")
print(f"{props.clockInstructionRate=}")
print(f"{props.clockRate=}")
print(f"{props.computeMode=}")
print(f"{props.concurrentKernels=}")
print(f"{props.concurrentManagedAccess=}")
print(f"{props.cooperativeLaunch=}")
print(f"{props.cooperativeMultiDeviceLaunch=}")
print(f"{props.cooperativeMultiDeviceUnmatchedBlockDim=}")
print(f"{props.cooperativeMultiDeviceUnmatchedFunc=}")
print(f"{props.cooperativeMultiDeviceUnmatchedGridDim=}")
print(f"{props.cooperativeMultiDeviceUnmatchedSharedMem=}")
print(f"{props.directManagedMemAccessFromHost=}")
print(f"{props.gcnArch=}")
print(f"{props.gcnArchName=}")
print(f"{props.integrated=}")
print(f"{props.isLargeBar=}")
print(f"{props.isMultiGpuBoard=}")
print(f"{props.kernelExecTimeoutEnabled=}")
print(f"{props.l2CacheSize=}")
print(f"{props.major=}")
print(f"{props.managedMemory=}")
print(f"{props.maxGridSize=}")
print(f"{props.maxSharedMemoryPerMultiProcessor=}")
print(f"{props.maxTexture1D=}")
print(f"{props.maxTexture1DLinear=}")
print(f"{props.maxTexture2D=}")
print(f"{props.maxTexture3D=}")
print(f"{props.maxThreadsDim=}")
print(f"{props.maxThreadsPerBlock=}")
print(f"{props.maxThreadsPerMultiProcessor=}")
print(f"{props.memPitch=}")
print(f"{props.memoryBusWidth=}")
print(f"{props.memoryClockRate=}")
print(f"{props.minor=}")
print(f"{props.multiProcessorCount=}")
print(f"{props.name=}")
print(f"{props.pageableMemoryAccess=}")
print(f"{props.pageableMemoryAccessUsesHostPageTables=}")
print(f"{props.pciBusID=}")
print(f"{props.pciDeviceID=}")
print(f"{props.pciDomainID=}")
print(f"{props.regsPerBlock=}")
print(f"{props.sharedMemPerBlock=}")
print(f"{props.tccDriver=}")
print(f"{props.textureAlignment=}")
print(f"{props.texturePitchAlignment=}")
print(f"{props.totalConstMem=}")
print(f"{props.totalGlobalMem=}")
print(f"{props.warpSize=}")
print("ok")