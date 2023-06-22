# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

# [literalinclude-begin]
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