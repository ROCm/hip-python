#!/usr/bin/env python3
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

"""Listing installed targets

This example shows how to list the installed targets
and obtain information such as their description.
"""

import sys

from rocm.bindings.llvm.c import core as _llvmc_core
from rocm.bindings.llvm.c import target as _llvmc_target

# The bindings load a shared LLVM at first call rather than at import, so an
# absent library would surface as a failed call deep in the example. Ask the
# bindings instead of inspecting the platform: has_symbol answers the capability
# question directly, and covers every reason the library may be missing --
# including any build configured with HIP_PYTHON_BUNDLE_LIBLLVM=OFF, which is
# the default on Windows because ROCm ships no shared LLVM there and one has to
# be linked from the static archives.
if not _llvmc_core.has_symbol("LLVMCreateMemoryBufferWithContentsOfFile"):
    raise NotImplementedError(
        "This example needs a loadable shared LLVM behind the "
        "rocm.bindings.llvm.c bindings; none was found. ROCm ships no shared "
        "LLVM on Windows, where rocm-bindings-compiler bundles one only when "
        "built with HIP_PYTHON_BUNDLE_LIBLLVM=ON."
    )

# A loadable `libLLVM.so` still ships in two flavours:
#   - the aggregate linked from the static archives (when
#     `HIP_PYTHON_FORCE_BUILD_LIBLLVM=ON` at configure time, and always
#     on Windows, where ROCm ships no shared LLVM), which exports the
#     `LLVMInitializeAll*` wrappers; and
#   - a copy of the system `libLLVM.so` (when the system library is
#     present and `HIP_PYTHON_FORCE_BUILD_LIBLLVM=OFF` — the default),
#     which is stripped of `LLVMInitializeAll*` because those wrappers
#     are `static inline` in `<llvm-c/Target.h>` and only surface in
#     the static archives.
# This example needs the `LLVMInitializeAll*` entry points; if the
# bundled libLLVM is the stripped system copy, print a notice and let
# the module return cleanly. (A bare `SystemExit` is treated as a
# failure by pytest's `runpy.run_path(...)` harness, but a normal
# module return is recorded as a pass.)
if not _llvmc_target.has_symbol("LLVMInitializeAllTargetInfos"):
    print(
        "list_targets: skipped — bundled libLLVM lacks "
        "LLVMInitializeAll* (rebuild rocm-bindings-compiler with "
        "HIP_PYTHON_FORCE_BUILD_LIBLLVM=ON to get the static-archive "
        "aggregate that exports them).",
        file=sys.stderr,
    )
else:
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

    # [literalinclude-iterate-targets-begin]
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
            target_features = "+xnack"
        machine = LLVMCreateTargetMachine(
            target,
            LLVMGetDefaultTargetTriple(),
            "generic",
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
    # [literalinclude-iterate-targets-end]

    # [literalinclude-target-from-triple-begin]
    print("Getting target for 'amdgcn-amd-amdhsa':")
    (status, target, error) = LLVMGetTargetFromTriple("amdgcn-amd-amdhsa")
    if target:
        print(f"- {LLVMGetTargetName(target)}")
    # [literalinclude-target-from-triple-end]
