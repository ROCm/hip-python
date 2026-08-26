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

"""How to use LLVM IR builder and interpreter

In this example, we implement the LLVM equivalent of

```c
int sum(int a, int b) {
    return a + b
}
```

by first constructing the LLVM IR tree in memory,
and then executing the resulting LLVM IR function via the
LLVM Interpreter ("Execution Engine")
We supply operands that the user of this script
specifies via the command line.

Acknowledgements:

    This example is derived from https://github.com/paulsmith/getting-started-llvm-c-api/blob/master/sum.c ,
    which was placed into the public domain (https://github.com/paulsmith/getting-started-llvm-c-api/blob/master/COPYING).
"""

import argparse

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
import sys

from rocm.bindings.llvm.c.analysis import (
    LLVMVerifierFailureAction,
    LLVMVerifyModule,
)
from rocm.bindings.llvm.c.bitwriter import LLVMWriteBitcodeToFile
from rocm.bindings.llvm.c.core import (
    LLVMAddFunction,
    LLVMAppendBasicBlock,
    LLVMBuildAdd,
    LLVMBuildRet,
    LLVMCreateBuilder,
    LLVMDisposeBuilder,
    LLVMDisposeMessage,
    LLVMFunctionType,
    LLVMGetNamedFunction,
    LLVMGetParam,
    LLVMInt32Type,
    LLVMModuleCreateWithName,
    LLVMPositionBuilderAtEnd,
)
from rocm.bindings.llvm.c.executionengine import (
    LLVMCreateGenericValueOfInt,
    LLVMCreateInterpreterForModule,
    LLVMDisposeExecutionEngine,
    LLVMGenericValueToInt,
    LLVMLinkInInterpreter,
    LLVMRunFunction,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes `x + y` via the LLVM interpreter"
    )
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    args = parser.parse_args()
    x, y = args.x, args.y
    bitcode_path = "sum.bc"
else:
    # Loaded under pytest (via runpy.run_path), which passes no arguments of
    # its own; parsing the test runner's would abort the example. The bitcode
    # goes to a scratch directory so a test run leaves the tree as it found it.
    import os.path
    import tempfile

    x, y = 1, 2
    bitcode_path = os.path.join(tempfile.gettempdir(), "sum.bc")

# [literalinclude-build-module-begin]
# Build the code
builder = LLVMCreateBuilder()

mod = LLVMModuleCreateWithName("my_module")

param_types = [LLVMInt32Type(), LLVMInt32Type()]
ret_type = LLVMFunctionType(LLVMInt32Type(), param_types, 2, 0)
sumfn = LLVMAddFunction(mod, "sum", ret_type)

entry = LLVMAppendBasicBlock(sumfn, "entry")

LLVMPositionBuilderAtEnd(builder, entry)
tmp = LLVMBuildAdd(
    builder, LLVMGetParam(sumfn, 0), LLVMGetParam(sumfn, 1), "tmp"
)
LLVMBuildRet(builder, tmp)
# [literalinclude-build-module-end]

# [literalinclude-verify-module-begin]
# Verify
_, error = LLVMVerifyModule(
    mod, LLVMVerifierFailureAction.LLVMAbortProcessAction
)
if error:
    print(f"error: {error}", file=sys.stderr)
    LLVMDisposeMessage(error)
# [literalinclude-verify-module-end]

# [literalinclude-run-interpreter-begin]
# Ask for the interpreter by name. LLVMCreateExecutionEngineForModule takes
# whatever it can get, which is the interpreter only as long as no code
# generator has been registered; in a process where something already called
# LLVMInitializeAllTargets it picks a target instead and aborts on the first
# one that cannot emit machine code.
LLVMLinkInInterpreter()
status, engine, error = LLVMCreateInterpreterForModule(mod)
if status != 0:
    print("failed to create execution engine", file=sys.stderr)
    if error:
        print(f"error: {error}", file=sys.stderr)
        LLVMDisposeMessage(error)
        sys.exit(1)

sumfn = LLVMGetNamedFunction(mod, "sum")
parms = [
    LLVMCreateGenericValueOfInt(LLVMInt32Type(), x, 0),
    LLVMCreateGenericValueOfInt(LLVMInt32Type(), y, 0),
]
res = LLVMRunFunction(engine, sumfn, 2, parms)
print(f"{LLVMGenericValueToInt(res, 0)}")
# [literalinclude-run-interpreter-end]

# Out to file
if LLVMWriteBitcodeToFile(mod, bitcode_path) != 0:
    print(
        f"error while writing file '{bitcode_path}', skipping",
        file=sys.stderr,
    )

# [literalinclude-dispose-begin]
# shutdown
LLVMDisposeExecutionEngine(engine)
# LLVMDisposeModule(mod) # TODO you can either call this or the above.
# Otherwise you get a segfault, investigate further.
LLVMDisposeBuilder(builder)
# [literalinclude-dispose-end]
