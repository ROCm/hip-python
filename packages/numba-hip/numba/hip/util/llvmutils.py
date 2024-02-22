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
    LLVMGetGlobalContext,
    LLVMCloneModule,
    LLVMModuleCreateWithName,
    LLVMIsDeclaration,
    LLVMGetValueName2,
    LLVMGetFirstFunction,
    LLVMGetNextFunction,
    LLVMDeleteFunction,
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
    if bc_len == None or bc_len < 1:
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

    Note:
        Always uses the global context.

    Args:
        ir (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM IR.
        ir_len (`int`, optional):
            Length of the LLVM IR buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(ir)``.
            Defaults to ``-1``.

    Returns:
        `tuple`:
            A `tuple` of size 5 that contains in that order:

            * status (`int`) - 0 means success.
            * mod - the parse result, the LLVM module
            * msg - an error message if status != 0.
            * ir_buf - LLVM buffer that the caller needs to dispose.
    """
    if isinstance(ir, str):
        ir = ir.encode("utf-8")
    if ir_len == None or ir_len < 1:
        ir_len = len(ir)

    buf = LLVMCreateMemoryBufferWithMemoryRange(
        ir,
        ir_len,
        b"llvm-ir-buffer",
        0,
    )
    # (status, mod, message, ir_llvm_buf)
    return (*LLVMParseIRInContext(LLVMGetGlobalContext(), buf), buf)


def _get_module(ir, ir_len: int = -1):
    """Load LLVM module from human-readable LLVM IR or LLVM bitcode.

    Args:
        ir (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM IR or LLVM BC.
        ir_len (`int`, optional):
            Length of the LLVM IR buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(ir)``.
            Defaults to ``-1``.
    Returns:
        `tuple`:
            A `tuple` ``(mod, bc_buf)`` that contains (in that order):

            * mod - The loaded LLVM module.
            * buf - An LLVM IR/BC buffer, or None.
            * from_bc - A flag indicating that the input is bitcode.
    """
    (parse_ir_status, mod, err_cstr, ir_buf) = _parse_llvm_ir(ir, ir_len)
    if parse_ir_status > 0:  # failure
        ir_err = err_cstr.decode("utf-8")
        LLVMDisposeMessage(err_cstr)
        LLVMDisposeModule(mod)
        LLVMDisposeMemoryBuffer(ir_buf)
        (parse_bc_status, mod, err_cstr, bc_buf) = _parse_llvm_bc(ir, ir_len)
        if parse_bc_status:
            bc_err = err_cstr.decode("utf-8")
            if err_cstr:
                LLVMDisposeMessage(err_cstr)
            raise ValueError(
                "input 'buf' seems to be neither (1) LLVM bitcode nor human-readable (2) LLVM IR."
                + f"\nReason (1): {bc_err}\nReason (2): {ir_err}"
            )
        return (mod, bc_buf, True)
    else:
        return (mod, None, False)


def _get_module_dispose_all(mod, buf, _):
    """Clean up the results of `_get_module`.

    Args:
        mod: A module.
        bc_buf: A bitcode buffer or None.
        context: A LLVM context that was used for parsing IR, or None.
    """
    LLVMDisposeModule(mod)
    if buf:
        LLVMDisposeMemoryBuffer(buf)


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
            Length of the LLVM IR buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(ir)``.
            Defaults to ``-1``.
    """
    (status, mod, msg, ir_llvm_buf) = _parse_llvm_ir(ir, ir_len)
    llvm_check(status, msg)  # disposes msg
    result = _to_bc(mod)
    LLVMDisposeModule(mod)
    LLVMDisposeMemoryBuffer(ir_llvm_buf)
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
            Length of the buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(mod)``.
            Defaults to ``-1``. Not used at all if ``mod`` is an instance of
            `rocm.llvm.c.types.LLVMOpaqueModule`.
    Returns:
        `bytes`:
            Always returns the resulting buffer as `bytes` object.
            Always returns a copy.
    """
    if isinstance(mod, LLVMOpaqueModule):
        return _to_ir(mod)
    else:
        gm_res = _get_module(mod, mod_len)
        result = _to_ir(mod=gm_res[0])
        _get_module_dispose_all(*gm_res)
        return result


def to_bc(mod, mod_len: int = -1):
    """Convert human-readable LLVM IR or LLVM bitcode to LLVM bitcode.

    Args:
        mod (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
            Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
        mod_len (`int`, optional):
            Length of the buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(mod)``.
            Defaults to ``-1``. Not used at all if ``mod`` is an instance of
            `rocm.llvm.c.types.LLVMOpaqueModule`.
    Returns:
        `bytes`:
            Always returns the resulting buffer as `bytes` object.
            Always returns a copy.
    """
    if isinstance(mod, LLVMOpaqueModule):
        return _to_bc(mod)
    else:
        gm_res = _get_module(mod, mod_len)
        result = _to_bc(mod=gm_res[0])
        _get_module_dispose_all(*gm_res)
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
            Length of the LLVM IR buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(mod)``.
            Defaults to ``-1``. Not used at all if ``mod`` is an instance of
            `rocm.llvm.c.types.LLVMOpaqueModule`.
    """
    if isinstance(mod, LLVMOpaqueModule):
        _verify(mod)
    else:
        gm_res = _get_module(mod, mod_len)
        _verify(mod=gm_res[0])
        _get_module_dispose_all(*gm_res)


def link_modules(
    modules, to_bc: bool = True, name: str = "link-modules-result"
) -> bytes:
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
                    Length of the LLVM IR buffer. Callers can specify numbers smaller than 1
                    or ``None`` to indicate that the buffer length should be derived via ``len(mod)``.
                    Defaults to ``-1``. Not used at all if ``mod`` is an instance of
                    `rocm.llvm.c.types.LLVMOpaqueModule`.
        to_bc (`bool`, optional):
            If the result should be LLVM bitcode instead of human-readable LLVM IR.
            Defaults to `True`.
        name (`str`, optional):
            Name for the resulting module.
    Returns:
        `bytes`:
            The result of the linking as LLVM bitcode or human-readable LLVM IR depending on argument ``to_bc``.
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
            gm_res = _get_module(ir, ir_len)
            cloned_modules.append(
                (LLVMCloneModule(gm_res[0]), gm_res)
            )  # store the result of _get_module to dispose later

    # LLVMLinkModules2(Dest, Src) "Links the source module into the destination module. The source module is destroyed."
    dest = LLVMModuleCreateWithName(name.encode("utf-8"))
    for src in reversed(cloned_modules):
        if LLVMLinkModules2(dest, src[0]) > 0:
            raise RuntimeError("An error has occurred")
    result = _to_bc(dest) if to_bc else _to_ir(dest)
    # clean up
    LLVMDisposeModule(dest)
    for _, gm_res in cloned_modules[:]:
        # the cloned modules have been consumed by the linker
        if gm_res:  # might be None if one input is instance of LLVMOpaqueModule
            _get_module_dispose_all(*gm_res)
    return result


def get_function_names(
    mod,
    mod_len: int = -1,
    matcher=lambda name: True,
    declares: bool = True,
    defines: bool = True,
):
    """Gets the names of matching functions in a module.

    Args:
        mod (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
            Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
        mod_len (`int`, optional):
            Length of the LLVM IR buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(mod)``.
            Defaults to ``-1``. Not used at all if ``mod`` is an instance of
            `rocm.llvm.c.types.LLVMOpaqueModule`.
        matcher (callable, optional):
            Function describing what a match is.
            Defaults to a lambda that returns ``True`` for any name.
        declares (`bool`, optional):
            Consider function declarations. Defaults to ``True``.
        defines (`bool`, optional):
            Consider function definitions. Defaults to ``True``.
    """

    def function_names_(mod):
        nonlocal declares
        nonlocal defines
        result = []
        fn = LLVMGetFirstFunction(mod)
        while fn:
            name_cstr, _ = LLVMGetValueName2(fn)
            if name_cstr:
                name = name_cstr.decode("utf-8")
                if matcher(name):
                    is_declare = LLVMIsDeclaration(fn) > 0
                    if is_declare and declares or not is_declare and defines:
                        result.append(name)
            fn = LLVMGetNextFunction(fn)
        return result

    if isinstance(mod, LLVMOpaqueModule):
        result = function_names_(mod)
    else:
        gm_res = _get_module(mod, mod_len)
        result = function_names_(mod=gm_res[0])
        _get_module_dispose_all(*gm_res)
    return result


def delete_functions(
    mod,
    mod_len: int = -1,
    matcher=lambda name: False,
    declares: bool = True,
    defines: bool = True,
):
    """Deletes all matching functions from a module.

    Note:
        With the default ``matcher``, no deletions are performed.

    Note:
        If the input is of type `rocm.llvm.c.types.LLVMOpaqueModule`,
        the passed in module is modified and returned.
        If the inputs are LLVM IR/BC buffers, the modified module
        is returned in the respective input form.

    Args:
        mod (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
            Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
        mod_len (`int`, optional):
            Length of the LLVM IR buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(mod)``.
            Defaults to ``-1``. Not used at all if ``mod`` is an instance of
            `rocm.llvm.c.types.LLVMOpaqueModule`.
        matcher (callable, optional):
            Function describing what a match is.
            Defaults to a lambda that returns ``False`` for any name.
        declares (`bool`, optional):
            Consider function declarations. Defaults to ``True``.
        defines (`bool`, optional):
            Consider function definitions. Defaults to ``True``.
    """

    def delete_functions_(mod):
        nonlocal matcher
        nonlocal declares
        nonlocal defines
        fn = LLVMGetFirstFunction(mod)
        while fn:
            name_cstr, _ = LLVMGetValueName2(fn)
            fn_next = LLVMGetNextFunction(fn)
            if name_cstr:
                name = name_cstr.decode("utf-8")
                if matcher(name):
                    is_declare = LLVMIsDeclaration(fn) > 0
                    if is_declare and declares or not is_declare and defines:
                        # print(f"delete function {name}")
                        LLVMDeleteFunction(fn)
            fn = fn_next

    if isinstance(mod, LLVMOpaqueModule):
        delete_functions_(mod)
        return mod
    else:
        (mod, buf, from_bc) = _get_module(mod, mod_len)
        delete_functions_(mod=mod)
        if from_bc:
            result = _to_bc(mod)
        else:
            result = _to_ir(mod)
        _get_module_dispose_all(mod, buf, from_bc)
        return result
