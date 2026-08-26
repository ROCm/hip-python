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

"""How to parse LLVM bitcode files

This example shows how to open and parse a user-supplied
bitcode file (via path). The example lists all function names and
the number of functions in the file.
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

from rocm.bindings.llvm.c.bitreader import LLVMParseBitcode2
from rocm.bindings.llvm.c.core import (
    LLVMCreateMemoryBufferWithContentsOfFile,
    LLVMDisposeMessage,
    LLVMGetFirstFunction,
    LLVMGetNextFunction,
    LLVMGetValueName2,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Counts number of functions found in the BC file and prints their name."
    )
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    filepath = args.path
else:
    # Loaded under pytest (via runpy.run_path) -- default to a known-good
    # bitcode file so the example doubles as an importable smoke test. The
    # device library sits directly under the ROCm root in a /opt/rocm-style
    # installation and under lib/llvm in the wheel and TheRock trees.
    import os
    import pathlib

    rocm_path = pathlib.Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    for _candidate in (
        rocm_path / "amdgcn" / "bitcode" / "opencl.bc",
        rocm_path / "lib" / "llvm" / "amdgcn" / "bitcode" / "opencl.bc",
    ):
        if _candidate.is_file():
            filepath = str(_candidate)
            break
    else:
        raise FileNotFoundError(
            f"no opencl.bc found under {rocm_path}; set ROCM_PATH to a ROCm "
            f"installation or pass a bitcode file on the command line"
        )


def check_status(status, message):
    """Reports an LLVM failure and stops.

    The calls below hand each other pointers that are only valid if the step
    before them succeeded, so continuing past a failure segfaults rather than
    reporting anything.
    """
    if status != 0:
        text = str(message)
        # LLVMParseBitcode2 reports no message of its own, so that call site
        # passes a plain string; only LLVM's own messages are LLVM's to free.
        if not isinstance(message, str):
            LLVMDisposeMessage(message)
        raise RuntimeError(text)


(status, buf, message) = LLVMCreateMemoryBufferWithContentsOfFile(filepath)
check_status(status, message)

(status, mod) = LLVMParseBitcode2(buf)
check_status(status, "failed to parse bitcode")

num_functions = 0
fn = LLVMGetFirstFunction(mod)  # a value type
while fn:
    name, name_len = LLVMGetValueName2(fn)
    if name:
        print(name)
    # TODO parse more information
    # https://marc.info/?l=llvm-dev&m=121146785917299&w=2
    # fn_type = LLVMTypeOf(fn)
    num_functions += 1
    fn = LLVMGetNextFunction(fn)
print(num_functions)
