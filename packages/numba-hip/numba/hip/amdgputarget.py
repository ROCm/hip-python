# MIT License
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

__author__ = "Advanced Micro Devices, Inc."

import logging
import threading

from rocm.llvm.c.core import LLVMDisposeMessage
from rocm.llvm.c.target import *
from rocm.llvm.c.targetmachine import *

from rocm.amd_comgr import amd_comgr as comgr


class AMDGPUTargetInitError(Exception):
    pass


logger = logging.getLogger(__name__)

# see: https://llvm.org/docs/AMDGPUUsage.html#address-spaces
ADDRSPACE_GENERIC = 0
ADDRSPACE_GLOBAL = 1
ADDRSPACE_SHARED = 3
ADDRSPACE_CONSTANT = 4
ADDRSPACE_LOCAL = 5

_lock = threading.Lock()


class ISAInfo:
    def __init__(self, info: dict):
        self._info = info

    @property
    def addressable_num_sgprs(self):
        return int(self._info["AddressableNumSGPRs"])

    @property
    def addressable_num_vgprs(self):
        return int(self._info["AddressableNumVGPRs"])

    @property
    def total_num_sgprs(self):
        return int(self._info["TotalNumSGPRs"])

    @property
    def total_num_vgprs(self):
        return int(self._info["TotalNumVGPRs"])

    @property
    def sgpr_alloc_granule(self):
        return int(self._info["SGPRAllocGranule"])

    @property
    def vgpr_alloc_granule(self):
        return int(self._info["VGPRAllocGranule"])

    @property
    def architecture(self):
        return self._info["Architecture"]

    @property
    def eus_per_cu(self):
        return int(self._info["EUsPerCU"])

    @property
    def environment(self):
        return self._info["Environment"]

    @property
    def features(self):
        return self._info["Features"]

    @property
    def lds_bank_count(self):
        return int(self._info["LDSBankCount"])

    @property
    def local_memory_size(self):
        return int(self._info["LocalMemorySize"])

    @property
    def max_flat_work_group_size(self):
        return int(self._info["MaxFlatWorkGroupSize"])

    @property
    def max_waves_per_cu(self):
        return int(self._info["MaxWavesPerCU"])

    @property
    def name(self):
        return self._info["Name"]

    @property
    def os(self):
        return self._info["OS"]

    @property
    def processor(self):
        return self._info["Processor"]

    @property
    def trap_handler_enabled(self):
        return int(self._info["TrapHandlerEnabled"]) > 0

    @property
    def vendor(self):
        return self._info["Vendor"]

    @property
    def version(self):
        return self._info["Version"]

    def __str__(self):
        return (
            f"<numba.hip.amdgpu.ISAInfo at {hex(id(self))}>(\n"
            + f"   {self.name=},\n"
            + f"   {self.architecture=},\n"
            + f"   {self.vendor=},\n"
            + f"   {self.os=},\n"
            + f"   {self.processor=},\n"
            + f"   {self.features=},\n"
            + f"   {self.environment=},\n"
            + f"   {self.max_flat_work_group_size=},\n"
            + f"   {self.max_waves_per_cu=},\n"
            + f"   {self.eus_per_cu=},\n"
            + f"   {self.addressable_num_sgprs=},\n"
            + f"   {self.addressable_num_vgprs=},\n"
            + f"   {self.total_num_sgprs=},\n"
            + f"   {self.total_num_vgprs=},\n"
            + f"   {self.sgpr_alloc_granule=},\n"
            + f"   {self.vgpr_alloc_granule=},\n"
            + f"   {self.local_memory_size=},\n"
            + f"   {self.lds_bank_count=},\n"
            + f"   {self.trap_handler_enabled=},\n"
            + f"   {self.version=}\n)\n"
        ).replace("self.", "")

    __repr__ = __str__


class AMDGPUTargetMachine:
    """Provides access to LLVM AMDGPU target machines for different AMD GPU ISAs.

    A singleton is created per pair of target architecture and target features.

    Class attributes:
        TRIPLE (`str`):
            The target triple, 'amdgcn-amd-amdhsa'.
        ISA_INFOS (`dict`):
            Per supported AMD ISA ('gfx...'), a data object
            that can be queried for information about hardware features
            such as the total/addressable number of SGPRs/VGPRs or the size
            of the LDS ('shared memory') as well as about metadata such
            as the vendor or the os.
    """

    TRIPLE = "amdgcn-amd-amdhsa"
    ISA_INFOS: dict = {
        isa_name.replace("amdgcn-amd-amdhsa--", ""): ISAInfo(entry)
        for isa_name, entry in comgr.ext.get_isa_metadata_all().items()
    }

    __INSTANCES = {}

    def __new__(cls, offload_arch: str, features: str = ""):
        with _lock:
            if not len(AMDGPUTargetMachine.__INSTANCES):
                logger.debug("[amdgpu] initialize LLVM target machines")
                LLVMInitializeAllTargetInfos()  # all three inits are required
                LLVMInitializeAllTargets()
                LLVMInitializeAllTargetMCs()
            arch_features = offload_arch + "--" + features
            if arch_features not in cls.__INSTANCES:
                cls.__INSTANCES[arch_features] = inst = object.__new__(cls)
        return cls.__INSTANCES[arch_features]

    def __init_target_machine(self, offload_arch: str, features: str = ""):
        logger.debug(
            f"[amdgpu] create LLVM AMDGPU target machine for arch-features pair '{offload_arch}')"
        )
        triple = self.TRIPLE.encode("utf-8")

        # create target
        (status, self.__target, error) = LLVMGetTargetFromTriple(triple)
        if status > 0:
            msg = str(error)
            LLVMDisposeMessage(error)
            raise AMDGPUTargetInitError(msg)

        # create target machine
        self.__keep_alive = (
            triple,
            offload_arch.split(":")[0].encode("utf-8"),  # remove feature part
            features.encode("utf-8"),
        )
        self.__target_machine = LLVMCreateTargetMachine(
            self.__target,
            *self.__keep_alive,
            LLVMCodeGenOptLevel.LLVMCodeGenLevelDefault,
            LLVMRelocMode.LLVMRelocDefault,
            LLVMCodeModel.LLVMCodeModelDefault,
        )
        data_layout = LLVMCreateTargetDataLayout(self.__target_machine)
        data_layout_cstr = LLVMCopyStringRepOfTargetData(data_layout)
        self.__data_layout = data_layout_cstr.decode("utf-8")
        LLVMDisposeMessage(data_layout_cstr)

    def __init__(self, offload_arch: str):
        self.__init_target_machine(offload_arch)

    @property
    def data_layout(self):
        return self.__data_layout

    def __del__(self):
        LLVMDisposeTargetMachine(self.__target_machine)


if __name__ in ("__main__", "__test__"):
    import pprint

    pprint.pprint(comgr.ext.get_isa_metadata_all())
    #pprint.pprint(AMDGPUTargetMachine.ISA_INFOS)
    machine = AMDGPUTargetMachine(offload_arch="gfx90a")
    print(machine.data_layout)
