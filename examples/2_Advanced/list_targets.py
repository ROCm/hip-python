#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

"""Listing installed targets

This example shows how to list the installed targets
and obtain information such as their description.
"""

from rocm.bindings.llvm.c.core import LLVMDisposeMessage
from rocm.bindings.llvm.c.target import (
    LLVMCopyStringRepOfTargetData,
    LLVMInitializeAllTargetInfos,
    LLVMInitializeAllTargetMCs,
    LLVMInitializeAllTargets,
)
from rocm.bindings.llvm.c.targetmachine import (
    LLVMCodeGenOptLevel,
    LLVMCodeModel,
    LLVMCreateTargetDataLayout,
    LLVMCreateTargetMachine,
    LLVMGetDefaultTargetTriple,
    LLVMGetFirstTarget,
    LLVMGetHostCPUFeatures,
    LLVMGetNextTarget,
    LLVMGetTargetFromTriple,
    LLVMGetTargetName,
    LLVMRelocMode,
)

print("List of installed targets:")
LLVMInitializeAllTargetInfos()  # all three inits are required
LLVMInitializeAllTargets()
LLVMInitializeAllTargetMCs()
target = LLVMGetFirstTarget()
while target:
    target_name = str(LLVMGetTargetName(target))
    print(f"- name: {target_name}")
    if target_name.startswith("x86"):
        target_features = LLVMGetHostCPUFeatures()
    else:
        target_features = b"+xnack"
    machine = LLVMCreateTargetMachine(
        target,
        LLVMGetDefaultTargetTriple(),
        b"generic",
        target_features,
        LLVMCodeGenOptLevel.LLVMCodeGenLevelDefault,
        LLVMRelocMode.LLVMRelocDefault,
        LLVMCodeModel.LLVMCodeModelDefault,
    )
    datalayout = LLVMCreateTargetDataLayout(machine)
    datalayout_str = LLVMCopyStringRepOfTargetData(datalayout)
    print(f"  data_layout: {datalayout_str}")
    LLVMDisposeMessage(datalayout_str)
    target = LLVMGetNextTarget(target)

print("Getting target for 'amdgcn-amd-amdhsa':")
(status, target, error) = LLVMGetTargetFromTriple(b"amdgcn-amd-amdhsa")
if target:
    print(f"- {LLVMGetTargetName(target)}")
