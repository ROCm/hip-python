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

"""Optimizing an LLVM module with the new pass manager

This example builds a small module whose function stores an
intermediate result to a stack slot, prints its LLVM IR, runs
the ``default<O2>`` pipeline over it via LLVM's new pass manager,
and prints the result. The stack slot disappears on the way.

The pipeline runs without a target machine, so nothing here is
AMD GPU specific; `numba.hip` uses the same three calls with a
target machine for ``amdgcn-amd-amdhsa``.
"""

# The bindings load a shared LLVM at first call rather than at import, so an
# absent library would surface as a failed call deep in the example. Ask the
# bindings instead of inspecting the platform: has_symbol answers the capability
# question directly, and covers every reason the library may be missing --
# including any build configured with HIP_PYTHON_BUNDLE_LIBLLVM=OFF, which is
# the default on Windows because ROCm ships no shared LLVM there and one has to
# be linked from the static archives.
from rocm.bindings.llvm.c import core as _llvmc_core

if not _llvmc_core.has_symbol("LLVMCreateMemoryBufferWithContentsOfFile"):
    raise NotImplementedError(
        "This example needs a loadable shared LLVM behind the "
        "rocm.bindings.llvm.c bindings; none was found. ROCm ships no shared "
        "LLVM on Windows, where rocm-bindings-compiler bundles one only when "
        "built with HIP_PYTHON_BUNDLE_LIBLLVM=ON."
    )

# [literalinclude-begin]
import argparse

from rocm.bindings.llvm.c.analysis import (
    LLVMVerifierFailureAction,
    LLVMVerifyModule,
)
from rocm.bindings.llvm.c.core import (
    LLVMAddFunction,
    LLVMAppendBasicBlockInContext,
    LLVMBuildAdd,
    LLVMBuildAlloca,
    LLVMBuildLoad2,
    LLVMBuildMul,
    LLVMBuildRet,
    LLVMBuildStore,
    LLVMContextCreate,
    LLVMContextDispose,
    LLVMCreateBuilderInContext,
    LLVMDisposeBuilder,
    LLVMDisposeMessage,
    LLVMFunctionType,
    LLVMGetParam,
    LLVMInt32TypeInContext,
    LLVMModuleCreateWithNameInContext,
    LLVMPositionBuilderAtEnd,
    LLVMPrintModuleToString,
)
from rocm.bindings.llvm.c.error import (
    LLVMDisposeErrorMessage,
    LLVMGetErrorMessage,
)
from rocm.bindings.llvm.c.transforms.passbuilder import (
    LLVMCreatePassBuilderOptions,
    LLVMDisposePassBuilderOptions,
    LLVMRunPasses,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs an LLVM pass pipeline over a generated module."
    )
    parser.add_argument(
        "--passes",
        type=str,
        default="default<O2>",
        help="pass pipeline in opt's -passes syntax",
    )
    passes = parser.parse_args().passes
else:
    # Loaded under pytest (via runpy.run_path), which passes no arguments of
    # its own; parsing the test runner's would abort the example.
    passes = "default<O2>"


# [literalinclude-print-module-begin]
def print_module(mod, headline):
    """Prints a module's LLVM IR.

    LLVMPrintModuleToString hands out a buffer that the caller owns, so
    the CStr must go back to LLVMDisposeMessage once it has been read.
    """
    ir = LLVMPrintModuleToString(mod)
    print(f"--- {headline} ---")
    print(str(ir), end="")
    LLVMDisposeMessage(ir)
    # [literalinclude-print-module-end]


# [literalinclude-build-module-begin]
# A context owns everything created in it, so disposing it at the end
# releases the module too -- calling LLVMDisposeModule as well would
# free the module twice.
context = LLVMContextCreate()
builder = LLVMCreateBuilderInContext(context)
mod = LLVMModuleCreateWithNameInContext("square_of_sum", context)

int32 = LLVMInt32TypeInContext(context)
fn_type = LLVMFunctionType(int32, [int32, int32], 2, 0)
fn = LLVMAddFunction(mod, "square_of_sum", fn_type)

entry = LLVMAppendBasicBlockInContext(context, fn, "entry")
LLVMPositionBuilderAtEnd(builder, entry)

# int slot = a + b; return slot * slot;
#
# The stack slot is what makes the pipeline's work visible: an
# unoptimized module keeps the alloca/store/load, and 'default<O2>'
# promotes it to a register.
slot = LLVMBuildAlloca(builder, int32, "slot")
total = LLVMBuildAdd(builder, LLVMGetParam(fn, 0), LLVMGetParam(fn, 1), "sum")
LLVMBuildStore(builder, total, slot)
reloaded = LLVMBuildLoad2(builder, int32, slot, "reloaded")
LLVMBuildRet(builder, LLVMBuildMul(builder, reloaded, reloaded, "square"))
# [literalinclude-build-module-end]

print_module(mod, "before")

# Verify before optimizing. A 'verify' pass, or any pipeline run with the
# VerifyEach option, reports a broken module by aborting the process, which
# no caller can catch; LLVMReturnStatusAction reports the same defect as a
# return value instead.
status, message = LLVMVerifyModule(
    mod, LLVMVerifierFailureAction.LLVMReturnStatusAction
)
if status != 0:
    text = str(message)
    LLVMDisposeMessage(message)
    raise RuntimeError(f"module is invalid: {text}")

# [literalinclude-run-passes-begin]
# The options object carries the pipeline's knobs (LLVMPassBuilderOptionsSet*);
# the defaults are what opt uses. The target machine is optional -- passing
# None hands LLVM a NULL LLVMTargetMachineRef, which selects a
# target-independent pipeline.
options = LLVMCreatePassBuilderOptions()
try:
    error = LLVMRunPasses(mod, passes, None, options)
    if error:
        # LLVMGetErrorMessage consumes the error; the message it returns is
        # the caller's to dispose, and with a different function than the
        # LLVMDisposeMessage used for module strings.
        message = LLVMGetErrorMessage(error)
        text = str(message)
        LLVMDisposeErrorMessage(message)
        raise RuntimeError(f"running passes '{passes}' failed: {text}")
finally:
    LLVMDisposePassBuilderOptions(options)
# [literalinclude-run-passes-end]

print_module(mod, f"after '{passes}'")

LLVMDisposeBuilder(builder)
LLVMContextDispose(context)
