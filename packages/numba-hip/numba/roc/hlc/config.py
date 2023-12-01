import sys
import os

# where ROCM bitcode is installed
DEFAULT_ROCM_BC_PATH = '/opt/rocm/amdgcn/bitcode/'
DEFAULT_OCLC_ABI_VERSION = 500

ROCM_BC_PATH = os.environ.get("NUMBA_ROCM_BC_PATH", DEFAULT_ROCM_BC_PATH)
OCLC_ABI_VERSION = os.environ.get("NUMBA_ROCM_OCLC_ABI_VERSION", DEFAULT_OCLC_ABI_VERSION)

# 32-bit private, local, and region pointers. 64-bit global, constant and flat.
# See:
# https://github.com/RadeonOpenCompute/llvm-project/blob/703e02d7aaa7b26d6bd9d03d6e9016e3a4dc9aa9/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp#L527
# Alloc goes into addrspace(5) (private)
DATALAYOUT = {
  64: ("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
       #"-p7:160:256:256:32-p8:128:128" ; this part is not compatible with the ROCm 5.6.0 BC files
       "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:"
       "128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
       #"-G1-ni:7:8" # this tail is not accepted by llvmlite 0.36.x
       )
}

# The data layout used by the ROCm AMDGCN BC files.
# We replace the data layout string in the LLVM IR with this layout.
AMDGCN_BC_DATALAYOUT = {
  64: ("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
       #"-p7:160:256:256:32-p8:128:128" ; this part is not compatible with the ROCm 5.6.0 BC files
       "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:"
       "128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
       "-G1-ni:7" # :8" ; this tail is not compatible with the ROCm 5.6.0 BC files
       )
}

TRIPLE = "amdgcn-amd-amdhsa"