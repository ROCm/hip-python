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

# [literalinclude-begin]
import sys

from rocm.bindings.llvm.c.analysis import LLVMVerifierFailureAction, LLVMVerifyModule
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
    LLVMCreateExecutionEngineForModule,
    LLVMCreateGenericValueOfInt,
    LLVMDisposeExecutionEngine,
    LLVMGenericValueToInt,
    LLVMLinkInInterpreter,
    LLVMRunFunction,
)

if __name__ == "__test__":
    x, y = 1, 2
else:
    parser = argparse.ArgumentParser(
        description="Computes `x + y` via the LLVM interpreter"
    )
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    args = parser.parse_args()
    x, y = args.x, args.y

# Build the code
builder = LLVMCreateBuilder()

# Note that the LLVM C APIs will typically not copy
# bytes/strings. Hence the Python objects that you pass
# to them should not go out of scope during the lifetime
# of the respective LLVM C object.

mod = LLVMModuleCreateWithName(b"my_module")

param_types = [LLVMInt32Type(), LLVMInt32Type()]
ret_type = LLVMFunctionType(LLVMInt32Type(), param_types, 2, 0)
sumfn = LLVMAddFunction(mod, b"sum", ret_type)

entry = LLVMAppendBasicBlock(sumfn, b"entry")

LLVMPositionBuilderAtEnd(builder, entry)
tmp = LLVMBuildAdd(
    builder, LLVMGetParam(sumfn, 0), LLVMGetParam(sumfn, 1), b"tmp"
)
LLVMBuildRet(builder, tmp)

# Verify
_, error = LLVMVerifyModule(
    mod, LLVMVerifierFailureAction.LLVMAbortProcessAction
)
if error:
    print(f"error: {error}", file=sys.stderr)
    LLVMDisposeMessage(error)

# Interpret the code with arguments from CLI
LLVMLinkInInterpreter()
status, engine, error = LLVMCreateExecutionEngineForModule(mod)
if status != 0:
    print("failed to create execution engine", file=sys.stderr)
    if error:
        print(f"error: {error}", file=sys.stderr)
        LLVMDisposeMessage(error)
        sys.exit(1)

sumfn = LLVMGetNamedFunction(mod, b"sum")
parms = [
    LLVMCreateGenericValueOfInt(LLVMInt32Type(), x, 0),
    LLVMCreateGenericValueOfInt(LLVMInt32Type(), y, 0),
]
res = LLVMRunFunction(engine, sumfn, 2, parms)
print(f"{LLVMGenericValueToInt(res, 0)}")

# Out to file
if LLVMWriteBitcodeToFile(mod, b"sum.bc") != 0:
    print("error while writing file 'sum.bc', skipping", file=sys.stderr)

# shutdown
LLVMDisposeExecutionEngine(engine)
# LLVMDisposeModule(mod) # TODO you can either call this or the above.
# Otherwise you get a segfault, investigate further.
LLVMDisposeBuilder(builder)
