# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

"""LLVM Utilities

This module contains tools for working with LLVM IR files
in human-readable and bitcode format.
"""

import copy

from rocm.llvm.c.core import (
    LLVMCreateMemoryBufferWithMemoryRange,
    LLVMDisposeMemoryBuffer,
    LLVMGetBufferSize,
    LLVMGetBufferStart,
    LLVMDisposeModule,
    LLVMPrintModuleToString,
    LLVMDisposeMessage,
    LLVMContextCreate,
    LLVMContextDispose,
)
from rocm.llvm.c.bitreader import LLVMParseBitcode2
from rocm.llvm.c.bitwriter import LLVMWriteBitcodeToMemoryBuffer
from rocm.llvm.c.irreader import LLVMParseIRInContext


def llvm_check(status, message):
    if status != 0:
        if isinstance(message, str):
            msg_str = message
        else:
            msg_str = str(message)
            LLVMDisposeMessage(message)
        raise RuntimeError(f"{msg_str}")


def convert_llvm_bc_to_ir(bcbuf, bcbuf_len: int = -1):
    """Convert LLVM bitcode to human-readable LLVM IR.

    Args:
        bcbuf (implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM BC.
        bcbuf_len (`int`):
            Length of the LLVM BC buffer.
    """
    if isinstance(bcbuf,str):
        bcbuf = bcbuf.encode("utf-8")
    if bcbuf_len < 1:
        bcbuf_len = len(bcbuf)

    buf = LLVMCreateMemoryBufferWithMemoryRange(
        bcbuf,
        bcbuf_len,
        b"llvm-ir-buffer",
        0,
    )
    (status, mod) = LLVMParseBitcode2(buf)
    llvm_check(status, "failed to parse bitcode")
    ir = LLVMPrintModuleToString(mod)
    result = copy.deepcopy(bytes(ir))  # copies into new buffer
    LLVMDisposeMessage(ir)
    LLVMDisposeModule(mod)
    LLVMDisposeMemoryBuffer(buf)
    return result


def convert_llvm_ir_to_bc(irbuf, irbuf_len: int=-1):
    """Convert human-readable LLVM IR to bitcode.

    Args:
        irbuf (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM IR.
        irbuf_len (`int`, optional):
            Length of the LLVM IR buffer. Must be supplied if it cannot
            be obtained via ``len(irbuf)``. 
    """
    if isinstance(irbuf,str):
        irbuf = irbuf.encode("utf-8")
    if irbuf_len < 1:
        irbuf_len = len(irbuf)

    ir_llvm_buf = LLVMCreateMemoryBufferWithMemoryRange(
        irbuf,
        irbuf_len,
        b"llvm-ir-buffer",
        0,
    )
    context = LLVMContextCreate()
    (status, mod, message) = LLVMParseIRInContext(context, ir_llvm_buf)
    llvm_check(status, message)
    bc_buf = LLVMWriteBitcodeToMemoryBuffer(mod)
    bc_llvm_buf_len = LLVMGetBufferSize(bc_buf)
    bc_llvm_buf = LLVMGetBufferStart(bc_buf).configure(
        _force=True, shape=(bc_llvm_buf_len,)
    )
    result = copy.deepcopy(bytes(bc_llvm_buf))  # copies into new buffer
    LLVMDisposeModule(mod)
    LLVMContextDispose(context)
    # LLVMDisposeMemoryBuffer(ir_buf) # TODO LLVMParseIRInContext seems to take ownership of the buffer and its deletion, memory analysis needed
    LLVMDisposeMemoryBuffer(bc_buf)
    return result
