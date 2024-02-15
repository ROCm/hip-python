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

import sys
import copy

from rocm.llvm.c.types import LLVMOpaqueModule
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
    LLVMCloneModule,
    LLVMModuleCreateWithName,
)
from rocm.llvm.c.bitreader import LLVMParseBitcode
from rocm.llvm.c.bitwriter import LLVMWriteBitcodeToMemoryBuffer
from rocm.llvm.c.irreader import LLVMParseIRInContext
from rocm.llvm.c.linker import LLVMLinkModules2
from rocm.llvm.c.analysis import (
    LLVMVerifyModule,
    LLVMVerifierFailureAction,
)


def llvm_check(status, message):
    """
    Note:
        Disposes message if status != 0 and it is a `~.CStr`.
    """
    if status != 0:
        if isinstance(message, str):
            msg_str = message
        else:
            msg_str = str(message)
            LLVMDisposeMessage(message)
        raise RuntimeError(f"{msg_str}")


def _parse_llvm_bc(bc, bc_len: int = -1):
    """Parse LLVM bitcode.

    Args:
        bc (implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM BC.
        bc_len (`int`):
            Length of the LLVM BC buffer.

    Returns:
        `tuple`:
            A `tuple` of size 4 that contains in that order:

            * status (`int`) - 0 means success.
            * mod - the parse result, the LLVM module
            * message - an error message if status != 0.
            * buf - LLVM buffer that the caller needs to dispose.
    """
    if isinstance(bc, str):
        bc = bc.encode("utf-8")
    if bc_len < 1:
        bc_len = len(bc)

    buf = LLVMCreateMemoryBufferWithMemoryRange(
        bc,
        bc_len,
        b"llvm-ir-buffer",
        0,
    )
    # (status, mod, msg, buf)
    return (*LLVMParseBitcode(buf), buf)


def _parse_llvm_ir(ir, ir_len: int = -1):
    """Parse human-readable LLVM IR.

    Args:
        ir (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM IR.
        ir_len (`int`, optional):
            Length of the LLVM IR buffer. Must be supplied if it cannot
            be obtained via ``len(irbuf)``.

    Returns:
        `tuple`:
            A `tuple` of size 5 that contains in that order:

            * status (`int`) - 0 means success.
            * mod - the parse result, the LLVM module
            * msg - an error message if status != 0.
            * ir_buf - LLVM buffer that the caller needs to dispose.
            * context - LLVM context that the caller needs to dispose.
    """
    if isinstance(ir, str):
        ir = ir.encode("utf-8")
    if ir_len < 1:
        ir_len = len(ir)

    ir_llvm_buf = LLVMCreateMemoryBufferWithMemoryRange(
        ir,
        ir_len,
        b"llvm-ir-buffer",
        0,
    )
    context = LLVMContextCreate()
    # (status, mod, message, ir_llvm_buf, context)
    return (*LLVMParseIRInContext(context, ir_llvm_buf), ir_llvm_buf, context)


def _get_module(ir, ir_len: int = -1):
    """Load LLVM module from human-readable LLVM IR or LLVM bitcode/

    Args:
        ir (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM IR or LLVM BC.
        ir_len (`int`, optional):
            Length of the LLVM IR/BC buffer. Must be supplied if it cannot
            be obtained via ``len(buf)``.
    Returns:
        `tuple`:
            A 3-`tuple` ``(mod, bc_buf, context)`` that contains (in that order):

            * mod - The loaded LLVM module.
            * bc_buf - A bitcode buffer, or None.
            * context - A LLVM context that was used for parsing IR, or None.
    """
    (parse_ir_status, mod, msg, ir_buf, context) = _parse_llvm_ir(ir, ir_len)
    if parse_ir_status > 0:  # failure
        LLVMDisposeMessage(msg)
        LLVMDisposeModule(mod)
        # LLVMDisposeMemoryBuffer(ir_buf) # TODO LLVMParseIRInContext seems to take ownership of the buffer and its deletion, memory analysis needed
        (parse_bc_status, mod, msg, bc_buf) = _parse_llvm_bc(ir, ir_len)
        if parse_bc_status:
            if msg:
                LLVMDisposeMessage(msg)
            raise ValueError(
                f"input 'buf' seems to be neither LLVM bitcode nor human-readable LLVM IR"
            )
    else:
        bc_buf = None
    return (mod, bc_buf, context)


def _get_module_dispose_all(mod, bc_buf, context):
    """Clean up the results of `_get_module`.

    Args:
        mod: A module.
        bc_buf: A bitcode buffer or None.
        context: A LLVM context that was used for parsing IR, or None.
    """
    LLVMDisposeModule(mod)
    if context:  # implies input is an IR module
        # LLVMDisposeMemoryBuffer(ir_buf) # TODO LLVMParseIRInContext seems to take ownership of the buffer and its deletion, memory analysis needed
        LLVMContextDispose(context)
    if bc_buf:  # implies input is an BC module
        LLVMDisposeMemoryBuffer(bc_buf)


def _print_module(mod: LLVMOpaqueModule):
    """Print llvm module to IR; mainly for debugging" """
    msg = LLVMPrintModuleToString(mod)
    print(msg)
    LLVMDisposeMessage(msg)


def _to_ir(mod: LLVMOpaqueModule):
    """Convert this LLVM Module to IR, return a copy."""
    ir = LLVMPrintModuleToString(mod)
    result = copy.deepcopy(bytes(ir))  # copies into new buffer
    LLVMDisposeMessage(ir)
    return result


def _to_bc(mod: LLVMOpaqueModule):
    """Convert this LLVM Module to IR, return a copy."""
    bc_buf = LLVMWriteBitcodeToMemoryBuffer(mod)
    bc_buf_len = LLVMGetBufferSize(bc_buf)
    bc_ndbuffer = LLVMGetBufferStart(bc_buf).configure(_force=True, shape=(bc_buf_len,))
    result = copy.deepcopy(bytes(bc_ndbuffer))  # copies into new buffer
    LLVMDisposeMemoryBuffer(bc_buf)
    return result


def to_ir_from_bc(bc, bc_len: int = -1):
    """Convert LLVM bitcode to human-readable LLVM IR.

    Args:
        bcbuf (implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM BC.
        bcbuf_len (`int`):
            Length of the LLVM BC buffer.
    """
    (status, mod, msg, buf) = _parse_llvm_bc(bc, bc_len)
    llvm_check(status, msg)
    result = _to_ir(mod)
    LLVMDisposeModule(mod)
    LLVMDisposeMemoryBuffer(buf)
    return result


def to_bc_from_ir(ir, ir_len: int = -1):
    """Convert human-readable LLVM IR to bitcode.

    Args:
        ir (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM IR.
        ir_len (`int`, optional):
            Length of the LLVM IR buffer. Must be supplied if it cannot
            be obtained via ``len(irbuf)``.
    """
    (status, mod, msg, ir_llvm_buf, context) = _parse_llvm_ir(ir, ir_len)
    llvm_check(status, msg)  # disposes msg
    result = _to_bc(mod)
    LLVMDisposeModule(mod)
    LLVMContextDispose(context)
    # LLVMDisposeMemoryBuffer(ir_llvm_buf) # TODO LLVMParseIRInContext seems to take ownership of the buffer and its deletion, memory analysis needed
    return result


def to_ir(mod, mod_len: int = -1):
    """Convert human-readable LLVM IR or LLVM bitcode to human-readable LLVM IR.

    Note:
        If the input is LLVM IR, this routine parses it and prints
        the resulting module to string (`bytes` to be exact).
        Hence, The result might look differently to the original input.

    Args:
        mod (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
            Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
        mod_len (`int`, optional):
            Length of the LLVM IR/BC buffer. Must be supplied if it cannot
            be obtained via ``len(mod)``. Not used at all if ``mod`` is no instance of
            `rocm.llvm.c.types.LLVMOpaqueModule`.
    Returns:
        `bytes`:
            Always returns the resulting buffer as `bytes` object.
            Always returns a copy.
    """
    if isinstance(mod, LLVMOpaqueModule):
        return _to_ir(mod)
    else:
        (mod, bc_buf, context) = _get_module(mod, mod_len)
        result = _to_ir(mod)
        _get_module_dispose_all(mod, bc_buf, context)
        return result


def to_bc(mod, mod_len: int = -1):
    """Convert human-readable LLVM IR or LLVM bitcode to LLVM bitcode.

    Note:
        If the input is LLVM IR, this routine parses it, prints
        the resulting module to string (`bytes` to be exact).
        The result might look differently to the original input.

    Args:
        mod (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
            Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
        mod_len (`int`, optional):
            Length of the LLVM IR/BC buffer. Must be supplied if it cannot
            be obtained via ``len(mod)``. Not used at all if ``mod`` is no instance of
            `rocm.llvm.c.types.LLVMOpaqueModule`.
    Returns:
        `bytes`:
            Always returns the resulting buffer as `bytes` object.
            Always returns a copy.
    """
    if isinstance(mod, LLVMOpaqueModule):
        return _to_bc(mod)
    else:
        (mod, bc_buf, context) = _get_module(mod, mod_len)
        result = _to_bc(mod)
        _get_module_dispose_all(mod, bc_buf, context)
        return result


def _verify(mod: LLVMOpaqueModule):
    """Raises `RuntimeError` if there are issues within the module."""
    retcode, err_cstr = LLVMVerifyModule(
        mod, LLVMVerifierFailureAction.LLVMReturnStatusAction
    )
    if retcode:
        if err_cstr:
            err = err_cstr.decode("utf-8")
            LLVMDisposeMessage(err_cstr)
            raise RuntimeError(err)
        else:
            raise RuntimeError()


def verify(mod, mod_len: int = -1):
    """Verifies the contents of an LLVM module.

    Args:
        mod (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
            Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
        mod_len (`int`, optional):
            Length of the LLVM IR/BC buffer. Must be supplied if it cannot
            be obtained via ``len(mod)``. Not used at all if ``mod`` is no instance of
            `rocm.llvm.c.types.LLVMOpaqueModule`.
    """
    if isinstance(mod, LLVMOpaqueModule):
        _verify(mod)
    else:
        (mod, bc_buf, context) = _get_module(mod, mod_len)
        _verify(mod)
        _get_module_dispose_all(mod, bc_buf, context)


def link_modules(modules, to_bc: bool = False):
    """Links the LLVM modules in the list together.

    Note:
        The result of this operation is order dependent.
        In this implementation, we create an empty module
        and then link it with ``modules[-1]`, ``modules[-2]``,
        ... ``modules[0]``, i.e. the specified modules are
        linked in reverse order.

    Args:
        modules (`iterable`):
            The modules to link. The output
            A list that contains entries of the following kind:

            1. Instance of `rocm.llvm.c.types.LLVMOpaqueModule`:
                ROCm LLVM Python module type.
            2. ir (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
                Buffer that contains LLVM IR or LLVM BC.
                Buffer size must be obtainable via `len(...)`.
            3. or a `tuple` that contains:
              * ir (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
                  Buffer that contains LLVM IR or LLVM BC.
              * ir_len (`int`):
                  Length of the LLVM IR/BC buffer. Must be supplied if it cannot
                  be obtained via ``len(buf)``.

        to_bc (`bool`, optional):
            If the result should be LLVM bitcode instead of human-readable LLVM IR.
            Defaults to `False`.
    """
    if not len(modules):
        raise ValueError("argument 'modules' must have at least one entry")
    # create LLVM module from every input
    cloned_modules = []
    for entry in modules:
        if isinstance(entry, LLVMOpaqueModule):
            cloned_modules.append((entry, None))
        else:
            if isinstance(entry, tuple):
                ir = entry[0]
                ir_len = entry[1]
            else:
                ir = entry
                ir_len = len(entry)
            (mod, bc_buf, context) = _get_module(ir, ir_len)
            cloned_modules.append(
                (LLVMCloneModule(mod), (mod, bc_buf, context))
            )  # store the result of _get_module to dispose later

    # LLVMLinkModules2(Dest, Src) "Links the source module into the destination module. The source module is destroyed."
    dest = LLVMModuleCreateWithName(b"link-modules-result")
    for src in reversed(cloned_modules):
        if LLVMLinkModules2(dest, src[0]) > 0:
            raise RuntimeError("An error has occurred")
    result = _to_bc(dest) if to_bc else _to_ir(dest)
    # clean up
    LLVMDisposeModule(dest)
    for _, to_dispose in cloned_modules[:]:
        # the cloned modules have been consumed by the linker
        if to_dispose:  # might be None if one input is instance of LLVMOpaqueModule
            _get_module_dispose_all(*to_dispose)
    return result


if __name__ in ("__main__",):
    # TODO move into unit test
    import textwrap

    # Test 1
    llvm_ir = textwrap.dedent(
        """\
    ; ModuleID = 'llvm-ir-buffer'
    source_filename = "source.hip"
    target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
    target triple = "amdgcn-amd-amdhsa"
    """
    )

    # equivalent LLVM bitcode
    llvm_bc = b'BC\xc0\xde5\x14\x00\x00\x05\x00\x00\x00b\x0c0$MY\xbef\xbd\xfb\xb4O\x1b\xc8$D\x012\x05\x00!\x0c\x00\x00c\x01\x00\x00\x0b\x02!\x00\x02\x00\x00\x00\x16\x00\x00\x00\x07\x81#\x91A\xc8\x04I\x06\x1029\x92\x01\x84\x0c%\x05\x08\x19\x1e\x04\x8bb\x80\x04E\x02B\x92\x0bB$\x102\x148\x08\x18K\n2\x12\x88Hp\xc4!#D\x12\x87\x8c\x10A\x92\x02d\xc8\x08\xb1\x14 CF\x88 \xc9\x012\x12\x84\x18*(*\x901|\xb0\\\x91 \xc1\xc8\x00\x00\x00\x89 \x00\x00\x08\x00\x00\x00"f\x04\x10\xb2B\x82I\x10RB\x82I\x90q\xc2PH\n\t&A\xc6\x05B\x12&\x08\x82\x81\x00\x00\x13\xa2ph\x07r8\x87qp\x876\x08\x87v \x876\x08\x87v \x07t\x98\x87p\xd8\x10\x17\xe5\xd0\x06\xf0\xa0\x07v@\x07z`\x07t\xd0\x06\xf0\x10\x07z`\x07t\xa0\x07v@\x07m\x00\x0fr\xa0\x07s \x07z0\x07r\xd0\x06\xf00\x07z0\x07r\xa0\x07s \x07m\x00\x0ft\xa0\x07v@\x07z`\x07t\xd0\x06\xf0P\x07z0\x07r\xa0\x07s \x07m\x00\x0fv\xa0\x07s \x07z0\x07r\xd0\x06\xe9`\x07t\xa0\x07v@\x07m`\x0fq`\x07z\x10\x07v\xd0\x06\xf6 \x07t\xa0\x07s \x07m`\x0fs \x07z0\x07r\xd0\x06\xf6@\x07x\xa0\x07v@\x07m`\x0fy`\x07z\x10\x07r\x80\x07m`\x0fq\x90\x07r\xa0\x07rP\x07v\xd0\x06\xf6 \x07u`\x07z \x07u`\x07m`\x0fu\x10\x07r\xa0\x07u\x10\x07r\xd0\x06\xf6\x10\x07p \x07t\xa0\x07q\x00\x07r@\x07m`\x0fr\x00\x07t\x80\x07z \x07p@\x07x\xd0\x06\xee0\x07r\xa0\x07v@\x07m0\x0bs \x07m\x10\nu\xd0\x06\xa7\x10\x07m\xe0\x0e\xe9\xa0\x07w\xa0\x11\xc2\x90\x8a\xe4P\x91@\xf8\x07\xf2Hl\x10(\x8a\x16\x00\x00\x10\x8b\x01\x00\x00\xb9\x00\x00\x003\x08\x80\x1c\xc4\xe1\x1cf\x14\x01=\x88C8\x84\xc3\x8cB\x80\x07yx\x07s\x98q\x0c\xe6\x00\x0f\xed\x10\x0e\xf4\x80\x0e3\x0cB\x1e\xc2\xc1\x1d\xce\xa1\x1cf0\x05=\x88C8\x84\x83\x1b\xcc\x03=\xc8C=\x8c\x03=\xccx\x8ctp\x07{\x08\x07yH\x87pp\x07zp\x03vx\x87p \x87\x19\xcc\x11\x0e\xec\x90\x0e\xe10\x0fn0\x0f\xe3\xf0\x0e\xf0P\x0e3\x10\xc4\x1d\xde!\x1c\xd8!\x1d\xc2a\x1ef0\x89;\xbc\x83;\xd0C9\xb4\x03<\xbc\x83<\x84\x03;\xcc\xf0\x14v`\x07{h\x077h\x87rh\x077\x80\x87p\x90\x87p`\x07v(\x07v\xf8\x05vx\x87w\x80\x87_\x08\x87q\x18\x87r\x98\x87y\x98\x81,\xee\xf0\x0e\xee\xe0\x0e\xf5\xc0\x0e\xec0\x03b\xc8\xa1\x1c\xe4\xa1\x1c\xcc\xa1\x1c\xe4\xa1\x1c\xdca\x1c\xca!\x1c\xc4\x81\x1d\xcaa\x06\xd6\x90C9\xc8C9\x98C9\xc8C9\xb8\xc38\x94C8\x88\x03;\x94\xc3/\xbc\x83<\xfc\x82;\xd4\x03;\xb0\xc3\x0c\xc7i\x87pX\x87rp\x83th\x07x`\x87t\x18\x87t\xa0\x87\x19\xceS\x0f\xee\x00\x0f\xf2P\x0e\xe4\x90\x0e\xe3@\x0f\xe1 \x0e\xecP\x0e3 (\x1d\xdc\xc1\x1e\xc2A\x1e\xd2!\x1c\xdc\x81\x1e\xdc\xe0\x1c\xe4\xe1\x1d\xea\x01\x1ef\x18Q8\xb0C:\x9c\x83;\xccP$v`\x07{h\x077`\x87wx\x07x\x98QL\xf4\x90\x0f\xf0P\x0e3\x1ej\x1e\xcaa\x1c\xe8!\x1d\xde\xc1\x1d~\x01\x1e\xe4\xa1\x1c\xcc!\x1d\xf0a\x06T\x85\x838\xcc\xc3;\xb0C=\xd0C9\xfc\xc2<\xe4C;\x88\xc3;\xb0\xc3\x8c\xc5\n\x87y\x98\x87w\x18\x87t\x08\x07z(\x07r\x98\x81\\\xe3\x10\x0e\xec\xc0\x0e\xe5P\x0e\xf30#\xc1\xd2A\x1e\xe4\xe1\x17\xd8\xe1\x1d\xde\x01\x1efH\x19;\xb0\x83=\xb4\x83\x1b\x84\xc38\x8cC9\xcc\xc3<\xb8\xc19\xc8\xc3;\xd4\x03<\xccH\xb4q\x08\x07v`\x07q\x08\x87qX\x87\x19\xdb\xc6\x0e\xec`\x0f\xed\xe0\x06\xf0 \x0f\xe50\x0f\xe5 \x0f\xf6P\x0en\x10\x0e\xe30\x0e\xe50\x0f\xf3\xe0\x06\xe9\xe0\x0e\xe4P\x0e\xf80#\xe2\xeca\x1c\xc2\x81\x1d\xd8\xe1\x17\xec!\x1d\xe6!\x1d\xc4!\x1d\xd8!\x1d\xe8!\x1ff \x9d;\xbcC=\xb8\x039\x94\x839\xccX\xbcpp\x07wx\x07z\x08\x07zH\x87wp\x87\x19\xcb\xe7\x0e\xef0\x0f\xe1\xe0\x0e\xe9@\x0f\xe9\xa0\x0f\xe50\xc3\x01\x03s\xa8\x07w\x18\x87_\x98\x87pp\x87t\xa0\x87t\xd0\x87r\x98\x81\x84A9\xe0\xc38\xb0C=\x90C9\xcc@\xc4\xa0\x1d\xca\xa1\x1d\xe0A\x1e\xde\xc1\x1cf$c0\x0e\xe1\xc0\x0e\xec0\x0f\xe9@\x0f\xe50C!\x83u\x18\x07sH\x87_\xa0\x87|\x80\x87r\x98\xb1\x94\x01<\x8c\xc3<\x94\xc38\xd0C:\xbc\x83;\xcc\xc3\x8c\xc5\x0cH!\x15Ba\x1e\xe6!\x1d\xce\xc1\x1dR\x81\x14\x00\xa9\x18\x00\x00\'\x00\x00\x00\x0b\nr(\x87w\x80\x07zXp\x98C=\xb8\xc38\xb0C9\xd0\xc3\x82\xe6\x1c\xc6\xa1\r\xe8A\x1e\xc2\xc1\x1d\xe6!\x1d\xe8!\x1d\xde\xc1\x1d\x164\xe3`\x0e\xe7P\x0f\xe1 \x0f\xe4@\x0f\xe1 \x0f\xe7P\x0e\xf4\xb0\x80\x81\x07y(\x87p`\x07vx\x87q\x08\x07z(\x07rXp\x9c\xc38\xb4\x01;\xa4\x83=\x94\xc3\x02k\x1c\xd8!\x1c\xdc\xe1\x1c\xdc \x1c\xe4a\x1c\xdc \x1c\xe8\x81\x1e\xc2a\x1c\xd0\xa1\x1c\xc8a\x1c\xc2\x81\x1d\xd8a\xc1\x01\x0f\xf4 \x0f\xe1P\x0f\xf4\x80\x0e\x0b\x88u\x18\x07sH\x07\x00\x00\x00\x00\xd1\x10\x00\x00\x06\x00\x00\x00\x07\xcc<\xa4\x83;\x9c\x03;\x94\x03=\xa0\x83<\x94C8\x90\xc3\x01\x00\x00\x00q \x00\x00\x02\x00\x00\x002\x0e\x10"\x04\x00\x00\x00\x00\x00\x00\x00e\x0c\x00\x00\x19\x00\x00\x00\x12\x03\x94\xb8\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x002\x00\x00\x00L\x00\x00\x00\x01\x00\x00\x00X\x00\x00\x00\x00\x00\x00\x00X\x00\x00\x00\x00\x00\x00\x00X\x00\x00\x00\x00\x00\x00\x002\x00\x00\x00\x11\x00\x00\x00C\x00\x00\x00\n\x00\x00\x00M\x00\x00\x00\x00\x00\x00\x00X\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00]\x0c\x00\x00\x17\x00\x00\x00\x12\x03\x94\xad\x00\x00\x00\x0017.0.0git 7208e8d15fbf218deb74483ea8c549c67ca4985eamdgcn-amd-amdhsasource.hip\x00\x00\x00\x00\x00\x00\x00'

    verify(llvm_ir)
    verify(llvm_bc)

    print(to_ir(llvm_ir))
    print(to_ir(llvm_bc))

    print(to_bc(llvm_ir))
    print(to_bc(llvm_bc))

    # Test 2
    main_llvm_ir = textwrap.dedent(
        # ```
        # // main.hip:
        # include "hip/hip_runtime.h"
        #
        # __device__ void scale(float* arr, float scal);
        #
        # __global__ void mykernel(float* arr, float scal) {
        #    scale(arr,scal);
        # }
        # ```
        # hipcc -S -emit-llvm main.hip
        """\
        ; ModuleID = 'main.hip'
        source_filename = "main.hip"
        target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
        target triple = "amdgcn-amd-amdhsa"

        ; Function Attrs: convergent mustprogress norecurse nounwind
        define protected amdgpu_kernel void @_Z8mykernelPff(ptr addrspace(1) %0, float %1) local_unnamed_addr #0 {
          %3 = addrspacecast ptr addrspace(1) %0 to ptr
          tail call void @_Z5scalePff(ptr %3, float %1) #2
          ret void
        }

        ; Function Attrs: convergent nounwind
        declare hidden void @_Z5scalePff(ptr, float) local_unnamed_addr #1

        attributes #0 = { convergent mustprogress norecurse nounwind "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
        attributes #1 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
        attributes #2 = { convergent nounwind }

        !llvm.module.flags = !{!0, !1, !2, !3, !4}
        !opencl.ocl.version = !{!5}
        !llvm.ident = !{!6}

        !0 = !{i32 4, !"amdgpu_hostcall", i32 1}
        !1 = !{i32 1, !"amdgpu_code_object_version", i32 500}
        !2 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
        !3 = !{i32 1, !"wchar_size", i32 4}
        !4 = !{i32 8, !"PIC Level", i32 2}
        !5 = !{i32 2, i32 0}
        !6 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)"}
        """
    )

    dep_llvm_ir = textwrap.dedent(
        # LLVM IR extracted from
        # hipcc -S -emit-llvm -fgpu-rdc dep.cpp
        # ```
        # // dep.cpp:
        # #include "hip/hip_runtime.h"
        # __device__ void scale_op(float arr[], float factor) {
        #    arr[threadIdx.x] *= factor;
        # }
        # ```
        """\
        ; ModuleID = 'dep.hip'
        source_filename = "dep.hip"
        target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
        target triple = "amdgcn-amd-amdhsa"

        ; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
        define hidden void @_Z5scalePff(ptr nocapture %0, float %1) local_unnamed_addr #0 {
          %3 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !0, !noundef !1
          %4 = zext i32 %3 to i64
          %5 = getelementptr inbounds float, ptr %0, i64 %4
          %6 = load float, ptr %5, align 4, !tbaa !2
          %7 = fmul contract float %6, %1
          store float %7, ptr %5, align 4, !tbaa !2
          ret void
        }

        ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
        declare i32 @llvm.amdgcn.workitem.id.x() #1

        attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+cumode,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+sramecc,+wavefrontsize64,-xnack" }
        attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

        !0 = !{i32 0, i32 1024}
        !1 = !{}
        !2 = !{!3, !3, i64 0}
        !3 = !{!"float", !4, i64 0}
        !4 = !{!"omnipotent char", !5, i64 0}
        !5 = !{!"Simple C++ TBAA"}
        """
    )

    linked = link_modules([main_llvm_ir, dep_llvm_ir]).decode("utf-8")
    verify(linked)
    print(linked)
