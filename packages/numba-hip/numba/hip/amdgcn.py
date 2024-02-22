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

"""AMD GPU target information and target machine creation

This module provides a class `~.AMDGPUTargetMachine` for creating
an LLVM target machine for all supported AMD GPU architectures.
The supported AMD GPU architectures are the keys of the `~.ISA_INFOS`
`dict` attribute of this module. More information on the particular
architecture can be obtained via the corresponding `dict` value.

Attributes:
    TRIPLE (`str`):
        The target triple, 'amdgcn-amd-amdhsa'.
    ISA_INFOS (`dict`):
        Per supported AMD ISA ('gfx...'), a data object
        that can be queried for information about hardware features
        such as the total/addressable number of SGPRs/VGPRs or the size
        of the LDS ('shared memory') as well as about metadata such
        as the vendor or the os.
    DATA_LAYOUT (`str`):
        Default data layout. Currently obtained via target machine for AMD GPU arch 'gfx90a'.
"""

__author__ = "Advanced Micro Devices, Inc."

import logging
import threading
import multiprocessing as mp
import sys

from rocm.llvm.c.types import LLVMOpaqueModule
from rocm.llvm.c.core import (
    LLVMDisposeMessage,
    LLVMCloneModule,
)
from rocm.llvm.c.error import (
    LLVMGetErrorMessage,
    LLVMDisposeErrorMessage,
)
from rocm.llvm.c.bitwriter import LLVMWriteBitcodeToMemoryBuffer

from rocm.llvm.c.target import *
from rocm.llvm.c.targetmachine import *
from rocm.llvm.c.transforms import passbuilder

from rocm.amd_comgr import amd_comgr as comgr

from numba.hip.util import llvmutils

_log = logging.getLogger(__name__)

# see: https://llvm.org/docs/AMDGPUUsage.html#address-spaces
ADDRSPACE_GENERIC = 0
ADDRSPACE_GLOBAL = 1
ADDRSPACE_SHARED = 3
ADDRSPACE_CONSTANT = 4
ADDRSPACE_LOCAL = 5


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
            f"<numba.hip.amdgputargetmachine.ISAInfo at {hex(id(self))}>(\n"
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


TRIPLE = "amdgcn-amd-amdhsa"
ISA_INFOS: dict = {
    isa_name.replace(f"{TRIPLE}--", ""): ISAInfo(entry)
    for isa_name, entry in comgr.ext.get_isa_metadata_all().items()
}


class AMDGPUTargetInitError(Exception):
    pass


_lock = threading.Lock()


def _RUN_PASSES(M, P, TM, O):
    """
    Note:
        As of ROCm 6.0.0 and LLVM 17.0.0, LLVMRunPasses raises
        an abort signal, which prevents us to capture
        any error. Furthermore, this forces
        us to create a child process. We let the child process
        abort and let the main process report a runtime error.
    """
    err = passbuilder.LLVMRunPasses(M, P, TM, O)
    if err:
        # TODO dead code, never reached as LLVMRunPasses raises SIGABRT
        msg = LLVMGetErrorMessage(err)  # consumes the error
        err_str = f'error: {msg.decode("utf-8")}'  # copies the message
        LLVMDisposeErrorMessage(msg)
        raise RuntimeError(f"error: {err_str}")


class AMDGPUTargetMachine:
    """Provides access to LLVM AMDGPU target machines for different AMD GPU ISAs.

    A singleton is created per pair of target architecture and target features.
    """

    __INSTANCES = {}

    def __new__(cls, target_cpu: str, target_features: str = ""):
        """
        Args:
            target_cpu (`str`):
                AMD GPU architecture, e.g. ``gfx90a``.
            target_features (`str`):
                Features that should be enabled for the target.
        """
        global _lock
        with _lock:
            if not len(AMDGPUTargetMachine.__INSTANCES):
                _log.debug("[amdgpu] initialize LLVM target machines")
                LLVMInitializeAllTargetInfos()  # all three inits are required
                LLVMInitializeAllTargets()
                LLVMInitializeAllTargetMCs()
            target_ident = target_cpu
            if target_features:
                target_ident += +"--" + target_features.replace(" ", "")
            if target_ident not in cls.__INSTANCES:
                cls.__INSTANCES[target_ident] = object.__new__(cls)
        return cls.__INSTANCES[target_ident]

    def __init_target_machine(self, target_cpu: str, target_features: str = ""):
        global TRIPLE

        _log.debug(
            f"[amdgpu] create LLVM AMDGPU target machine for arch-features pair '{target_cpu}')"
        )
        triple = TRIPLE.encode("utf-8")

        # create target
        (status, self._target, error) = LLVMGetTargetFromTriple(triple)
        if status > 0:
            msg = str(error)
            LLVMDisposeMessage(error)
            raise AMDGPUTargetInitError(msg)

        # create target machine
        self._keep_alive = (
            triple,
            target_cpu.split(":")[0].encode("utf-8"),  # remove feature part
            target_features.encode("utf-8"),
        )
        self._target_machine = LLVMCreateTargetMachine(
            self._target,
            *self._keep_alive,
            LLVMCodeGenOptLevel.LLVMCodeGenLevelDefault,
            LLVMRelocMode.LLVMRelocDefault,
            LLVMCodeModel.LLVMCodeModelDefault,
        )
        data_layout = LLVMCreateTargetDataLayout(self._target_machine)
        data_layout_cstr = LLVMCopyStringRepOfTargetData(data_layout)
        self._data_layout = data_layout_cstr.decode("utf-8")
        LLVMDisposeMessage(data_layout_cstr)

    def __init__(self, offload_arch: str):
        if not hasattr(self, "_keep_alive"):  # already initialized
            self.__init_target_machine(offload_arch)

    @property
    def data_layout(self):
        return self._data_layout

    def optimize_module(
        self, mod, mod_len: int = -1, passes: str = "default<O3>", **pass_builder_opts
    ):
        r"""Optimizes LLVM IR, bitcode, or `rocm.llvm.c.types.LLVMOpaqueModule`.

        Args:
            mod (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
                Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
            mod_len (`int`, optional):
                Length of the LLVM IR/BC buffer. Must be supplied if it cannot
                be obtained via ``len(mod)``. Not used at all if ``mod`` is no instance of
                `rocm.llvm.c.types.LLVMOpaqueModule`.
            passes (UTF-8 `str`):
                The format of this string is the same as opt's -passes argument for the new pass
                manager. Individual passes may be specified, separated by commas. Full
                pipelines may also be invoked using ``default<O3>`` and friends. See opt for
                full reference of the ``passes`` format. Defaults to ``default<O3>``.
            \*\*pass_builder_opts (keyword arguments):
                Pairs of boolean values and keys such as ``VerifyEach`` that can be
                mapped directly to a pass builder option `LLVMPassBuilderOptionsSet<key>`.
                Note that the number of available C API tranforms changes frequently
                with LLVM releases. Hence, this approach to passing such options was chosen.
                The ``passes`` argument is more stable and should be preferred.

        Returns:
            The optimized module in the input format.
        """
        opts = passbuilder.LLVMCreatePassBuilderOptions()
        option_setter_prefix = "LLVMPassBuilderOptionsSet"
        for k, v in pass_builder_opts:
            try:
                setter = getattr(passbuilder, option_setter_prefix + k)
            except AttributeError:
                available_opts = ", ".join(
                    [
                        f'{k.replace(option_setter_prefix,"")}'
                        for k in vars(passbuilder).keys()
                        if k.startswith(option_setter_prefix)
                    ]
                )
                raise KeyError(
                    f"unknown pass builder option '{k}', use one of: {available_opts}"
                )
            else:
                if not isinstance(v, bool):
                    return ValueError(
                        "pass builder option values must be of type 'bool'"
                    )
                setter(opts, int(v))

        if isinstance(mod, LLVMOpaqueModule):
            optimized = mod
        else:
            gm_res = llvmutils._get_module(mod, mod_len)
            optimized = gm_res[0]
            from_bc = gm_res[-1]

        # As LLVMRunPasses aborts the process, we need to run it in a separate process
        # stderr_post = sys.stderr
        process = mp.Process(
            target=_RUN_PASSES,
            args=(optimized, passes.encode("utf-8"), self._target_machine, opts),
        )
        process.start()
        process.join()
        # The child’s exit code. This will be None if the process has not yet terminated.
        # If the child’s run() method returned normally, the exit code will be 0.
        # If it terminated via sys.exit() with an integer argument N, the exit code will be N.
        # If the child terminated due to an exception not caught within run(),
        # the exit code will be 1. If it was terminated by signal N, the exit code
        # will be the negative value -N.
        # https://docs.python.org/3.9/library/multiprocessing.html#multiprocessing.Process.exitcode
        if process.exitcode != 0:
            raise RuntimeError(
                "LLVMRunPasses failed and was aborted; please check error output"
            )

        if isinstance(mod, LLVMOpaqueModule):
            result = optimized
        else:
            result = (
                llvmutils.to_bc(optimized) if from_bc else llvmutils.to_ir(optimized)
            )

        # clean up
        if not isinstance(mod, LLVMOpaqueModule):
            llvmutils._get_module_dispose_all(*gm_res)
        passbuilder.LLVMDisposePassBuilderOptions(opts)
        return result

    def verify_module(self, ir, ir_len: int = -1):
        """Returns verified LLVM IR in the input format.

        Calls ``self.optimize_llvm_ir`` with ``passes='verify'``.

        Args:
            mod (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
                Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
            mod_len (`int`, optional):
                Length of the LLVM IR/BC buffer. Must be supplied if it cannot
                be obtained via ``len(mod)``. Not used at all if ``mod`` is no instance of
                `rocm.llvm.c.types.LLVMOpaqueModule`.
        See:
            `~.AMDGPUTargetMachine.optimize_llvm_ir`.
        Returns:
            The optimized module in the input format.
        """
        return self.optimize_module(ir, ir_len, passes="verify")

    def __del__(self):
        LLVMDisposeTargetMachine(self._target_machine)


# We define 'gfx90a' data layout as default data layout.
# No changes noticed.
DATA_LAYOUT = AMDGPUTargetMachine("gfx90a").data_layout

__all__ = [
    "ADDRSPACE_GENERIC",
    "ADDRSPACE_GLOBAL",
    "ADDRSPACE_SHARED",
    "ADDRSPACE_CONSTANT",
    "ADDRSPACE_LOCAL",
    "ISAInfo",
    "TRIPLE",
    "ISA_INFOS" "AMDGPUTargetInitError",
    "AMDGPUTargetMachine",
    "DATA_LAYOUT",
]
