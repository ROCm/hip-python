# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

# flake8: noqa

__author__ = "Advanced Micro Devices, Inc."

"""LLVM Utilities

This module contains tools for working with LLVM IR files
in human-readable and bitcode format.
"""

import copy
import re

from rocm.llvm.c.analysis import (
    LLVMVerifierFailureAction,
    LLVMVerifyModule,
)
from rocm.llvm.c.bitreader import LLVMGetBitcodeModuleInContext2
from rocm.llvm.c.bitwriter import LLVMWriteBitcodeToMemoryBuffer
from rocm.llvm.c.core import (  # LLVMDeleteFunction,; LLVMDeleteGlobal,
    LLVMAliasGetAliasee,
    LLVMCallConv,
    LLVMContextCreate,
    LLVMContextDispose,
    LLVMCreateMemoryBufferWithMemoryRange,
    LLVMDisposeMemoryBuffer,
    LLVMDisposeMessage,
    LLVMDisposeModule,
    LLVMGetBufferSize,
    LLVMGetBufferStart,
    LLVMGetCalledValue,
    LLVMGetFirstBasicBlock,
    LLVMGetFirstFunction,
    LLVMGetFirstGlobalAlias,
    LLVMGetFirstInstruction,
    LLVMGetFunctionCallConv,
    LLVMGetNextBasicBlock,
    LLVMGetNextFunction,
    LLVMGetNextGlobalAlias,
    LLVMGetNextInstruction,
    LLVMGetValueName2,
    LLVMGetVisibility,
    LLVMIsACallInst,
    LLVMIsAFunction,
    LLVMIsAGlobalAlias,
    LLVMIsAInlineAsm,
    LLVMIsDeclaration,
    LLVMLinkage,
    LLVMModuleCreateWithNameInContext,
    LLVMPrintModuleToString,
    LLVMPrintValueToString,
    LLVMSetLinkage,
    LLVMSetVisibility,
    LLVMVisibility,
)
from rocm.llvm.c.irreader import LLVMParseIRInContext
from rocm.llvm.c.linker import LLVMLinkModules2
from rocm.llvm.c.types import (
    LLVMOpaqueContext,
    LLVMOpaqueModule,
    LLVMOpaqueValue,
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


def _parse_llvm_bc_in_context(
    context: LLVMOpaqueContext, bc, bc_len: int = -1
):
    """Parse LLVM bitcode in the given context.

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
    if bc_len is None or bc_len < 1:
        bc_len = len(bc)

    buf = LLVMCreateMemoryBufferWithMemoryRange(
        bc,
        bc_len,
        b"llvm-bc-buffer",
        0,
    )
    # (err_status, module, err_msg, buf)
    res = LLVMGetBitcodeModuleInContext2(context, buf)
    err_msg = "failed to parse LLVM BC buffer" if res[0] > 0 else None
    return (*res, err_msg, buf)


def _parse_llvm_ir_in_context(context, ir, ir_len: int = -1):
    """Parse both human-readable LLVM IR or LLVM bitcode.

    Note:
        Both formats human-readable LLVM IR LLVM assembly and
        LLVM bitcode are supported by routine
        `rocm.llvm.c.irreader.LLVMParseIRInContext`
        which is called by this function.

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
    if ir_len is None or ir_len < 1:
        ir_len = len(ir)

    buf = LLVMCreateMemoryBufferWithMemoryRange(
        ir,
        ir_len,
        b"llvm-ir-buffer",
        0,
    )
    # (status, mod, message)
    result = LLVMParseIRInContext(context, buf)
    return result


def _get_module_in_context(context: LLVMOpaqueContext, ir, ir_len: int = -1):
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
            A `tuple` ``(mod, )`` that contains (in that order):

            * mod - The loaded LLVM module.
    Note:
        Return list might need to be extended,
        hence the tuple result.
    See:
        _get_module_dispose_all
    """
    (status, mod, err_cstr) = _parse_llvm_ir_in_context(context, ir, ir_len)
    if status > 0:  # failure
        errmsg = err_cstr.decode("utf-8")
        LLVMDisposeMessage(err_cstr)
        if mod:
            LLVMDisposeModule(mod)
        # LLVMDisposeMemoryBuffer(ir_buf) mod seems to take ownership of the buffer # TODO(HIP/AMD) check memory
        raise ValueError(
            "input 'buf' seems to be neither valid LLVM bitcode nor LLVM assembly.\n\n"
            f"Reason: {errmsg}"
        )
    else:
        return (mod,)


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
    bc_ndbuffer = LLVMGetBufferStart(bc_buf).configure(
        _force=True, shape=(bc_buf_len,)
    )
    result = copy.deepcopy(bytes(bc_ndbuffer))  # copies into new buffer
    LLVMDisposeMemoryBuffer(bc_buf)
    return result


def to_ir_from_bc(bc, bc_len: int = -1):
    """LLVM bitcode as humand-readable LLVM assembly.

    Args:
        bcbuf (implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM BC.
        bcbuf_len (`int`):
            Length of the LLVM BC buffer.
    """
    context = LLVMContextCreate()
    (status, mod, msg, _) = _parse_llvm_bc_in_context(context, bc, bc_len)
    llvm_check(status, msg)
    result = _to_ir(mod)
    LLVMDisposeModule(mod)
    # LLVMDisposeMemoryBuffer(_) # NOTE: module seems to have taken ownership
    LLVMContextDispose(context)
    return result


def to_bc_from_ir(ir, ir_len: int = -1):
    """Human-readable LLVM assembly or LLVM bitcode as LLVM bitcode.

    Args:
        ir (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
            Buffer that contains LLVM IR.
        ir_len (`int`, optional):
            Length of the LLVM IR buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(ir)``.
            Defaults to ``-1``.
    """
    context = LLVMContextCreate()
    (status, mod, msg) = _parse_llvm_ir_in_context(context, ir, ir_len)
    llvm_check(status, msg)  # disposes msg
    result = _to_bc(mod)
    LLVMDisposeModule(mod)
    # LLVMDisposeMemoryBuffer(ir_buf) mod seems to take ownership of the buffer # TODO(HIP/AMD) check memory ownership
    LLVMContextDispose(context)
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
        context = LLVMContextCreate()
        (mod,) = _get_module_in_context(context, mod, mod_len)
        result = _to_ir(mod)
        LLVMDisposeModule(mod)
        LLVMContextDispose(context)
        return result


def to_ir_fast(mod, mod_len: int = -1):
    """Convert human-readable LLVM IR or LLVM bitcode to human-readable LLVM IR.

    Fast version of `to_ir` that does not return a copy
    if the input is already human-readable LLVM assembly.
    This routines assumes that the input is human-readable
    IR if the first two bytes of the input 'mod' are not the
    ASCII chars "BC".

    Returns:
        `bytes`: Always returns the result as bytes.
    """
    try:
        if isinstance(mod, str):
            as_bytes = bytes(mod, encoding="utf-8")
        else:
            as_bytes = bytes(mod)
        if as_bytes[0:2] != b"BC":
            return as_bytes
    except TypeError:
        pass
    return to_ir(mod, mod_len)


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
        context = LLVMContextCreate()
        gm_res = _get_module_in_context(context, mod, mod_len)
        result = _to_bc(mod=gm_res[0])
        LLVMDisposeModule(*gm_res)
        LLVMContextDispose(context)
        return result


def to_bc_fast(mod, mod_len: int = -1):
    """Convert human-readable LLVM IR or LLVM bitcode to LLVM bitcode.

    Fast version of `to_bc` that does not return a copy
    if the input is already LLVM bitcode.
    This routines assumes that the input is LLVM bitcode
    if the first two bytes of the input 'mod' are the
    ASCII chars "BC".
    """
    try:
        if isinstance(mod, str):
            as_bytes = bytes(mod, encoding="utf-8")
        else:
            as_bytes = bytes(mod)
        if as_bytes[0:2] == b"BC":
            return as_bytes
    except TypeError:
        pass
    return to_bc(mod, mod_len)


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
        context = LLVMContextCreate()
        gm_res = _get_module_in_context(context, mod, mod_len)
        _verify(mod=gm_res[0])
        LLVMDisposeModule(*gm_res)
        LLVMContextDispose(context)


class LLVMModuleWrapper:
    """Wrapper class for handling LLVM modules.

    Stores all input formats in serialized form (LLVM IR or BC).
    """

    def __init__(self, mod, mod_len: int = -1):
        """LLVM module wrapper.

        Args:
            context (`rocm.llvm.c.tpyes.LLVMOpaqueContext`):
                The LLVM context to create modules in.
            mod (`rocm.llvm.c.types.LLVMOpaqueModule`, `LLVMModuleWrapper`, or UTF-8 `str`, or Python buffer like `bytes`):
                An 'rocm.llvm.c.types.LLVMOpaqueModule', `LLVMModuleWrapper`, or a buffer that contains LLVM IR or LLVM BC.
                If you pass an `rocm.llvm.c.types.LLVMOpaqueModule`, then
                code content is serialized to BC.
            mod_len (`int`, optional):
                Length of the LLVM IR/BC buffer. Callers can specify numbers smaller than 1
                or ``None`` to indicate that the buffer length should be derived via ``len(ir)``.
                Defaults to ``-1``.
        Raises:
            KeyError: _description_
        """
        if isinstance(mod, LLVMModuleWrapper):
            self._bc_or_ir = mod.bc_or_ir
            self._bc_or_ir_len = len(self.bc_or_ir)
        elif isinstance(mod, LLVMOpaqueModule):
            self._bc_or_ir = to_bc(mod)
            self._bc_or_ir_len = len(self._bc_or_ir)
        else:
            self._bc_or_ir = mod
            self._bc_or_ir_len = mod_len

    def create_mod_in_context(self, context):
        """Lazily creates LLVM module if not already available."""
        gm_res = _get_module_in_context(
            context, self._bc_or_ir, self._bc_or_ir_len
        )
        mod = gm_res[0]
        return mod

    @property
    def ir(self) -> bytes:
        """Return human-readable LLVM IR if not already available."""
        return to_ir_fast(self._bc_or_ir)

    @property
    def bc(self) -> bytes:
        """Return LLVM BC if not already available."""
        return to_bc_fast(self._bc_or_ir)

    @property
    def bc_or_ir(self) -> bytes:
        """Lazily produces LLVM BC if not already LLVM BC/IR available."""
        if not self._bc_or_ir:
            return self._bc_or_ir
        return to_bc_fast(self._bc_or_ir)

    def __str__(self):
        return self.ir.decode(encoding="utf-8")

    def __dealloc__(self):
        if self._owner and self._mod:
            LLVMDisposeModule(self._mod)


def link_modules(
    modules,
    to_bc: bool = True,
    name: str = "link-modules-result",
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
            2. Instance of `numba.hip.util.llvmutils.LLVMModuleWrapper`:
                 Numba HIP wrapper for ROCm LLVM Python modules.
            3. ir (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`):
                 Buffer that contains LLVM IR or LLVM BC.
                 Buffer size must be obtainable via `len(...)`.
            4. or a `tuple` that contains:
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
    linker_inputs = []
    context = LLVMContextCreate()
    for entry in modules:
        if isinstance(entry, (tuple, list)):
            wrapper = LLVMModuleWrapper(*entry)
        else:
            wrapper = LLVMModuleWrapper(entry)
        linker_inputs.append(wrapper.create_mod_in_context(context))
        # _verify(linker_inputs[-1])

    # LLVMLinkModules2(Dest, Src) "Links the source module into the destination module. The source module is destroyed."
    dest = LLVMModuleCreateWithNameInContext(name.encode(), context)
    for src in reversed(linker_inputs):
        if LLVMLinkModules2(dest, src) > 0:
            raise RuntimeError("An error has occurred")
    result = _to_bc(dest) if to_bc else _to_ir(dest)
    # clean up
    # print("result")
    # print(to_ir_fast(result))
    LLVMDisposeModule(dest)
    LLVMContextDispose(context)
    return result


def _llvm_value_name_as_str(llvm_value):  # type: (LLVMOpaqueValue) -> str
    """Gets LLVM value's name as Python 'str'."""
    name_cstr = LLVMGetValueName2(llvm_value)[0]
    if name_cstr:
        return bytes(name_cstr).decode()
    return None


def _iter_functions(mod):
    """Iterates all functions"""
    fn = LLVMGetFirstFunction(mod)  # a value type
    while fn:
        yield (fn, _llvm_value_name_as_str(fn))
        fn = LLVMGetNextFunction(fn)


def _iter_global_function_aliases(mod):
    """Iterates all global alias."""
    alias = LLVMGetFirstGlobalAlias(mod)
    while alias:
        yield (alias, _llvm_value_name_as_str(alias))
        alias = LLVMGetNextGlobalAlias(alias)


def _get_function_aliases(
    mod,
):  # type: (LLVMOpaqueModule) -> dict[str, LLVMOpaqueValue]
    """Finds and completely expands function aliases.

    Args:
        mod (`~.LLVMOpaqueModule`):
            An LLVM module.
    Returns:
        `dict` of `str` and `~.LLVMOpaqueValue`:
            Maps function alias name to fully expanded aliasee.
    """
    function_aliases = dict()

    def expand_aliases_(aliases, alias):
        nonlocal function_aliases
        aliasee = LLVMAliasGetAliasee(alias)
        aliasee_name = _llvm_value_name_as_str(aliasee)
        if LLVMIsAFunction(aliasee) is not None:
            for alias in aliases:
                function_aliases[alias] = aliasee
        elif LLVMIsAGlobalAlias(aliasee):
            expand_aliases_(aliases + [aliasee_name], aliasee)
        else:
            pass

    # Collect and expand aliases
    for alias, alias_name in _iter_global_function_aliases(mod):
        expand_aliases_([alias_name], alias)

    return function_aliases


def _get_functions_called_by_amdgpu_kernel_hide_device_functions(
    mod, function_aliases
):
    """

    Identifies functions that care called by a function
    with `amdgpu_kernel` (calling convention).
    The call might be done directly or indirectly via
    a global function alias.

    Further changes the visibility and linkage of functions with
    other calling convention so that they do not appear
    as symbols in generated machine code or AMD GPU HSA assembly.

    Warning:
        Side effects: Visibility and linkage of functions
        in input argument 'mod' is modified ('hidden', 'private')
        to exclude them from compiled code objects and
        generated HSA assembly.

    Change visibility if
    """
    used_functions = set()

    # now collect the used functions
    def identify_callees_(name, fn):
        nonlocal used_functions
        used_functions.add(name)  # implicit copy
        if LLVMIsDeclaration(fn) > 0:
            return
        bb = LLVMGetFirstBasicBlock(fn)
        while bb:
            instr = LLVMGetFirstInstruction(bb)
            while instr:
                if LLVMIsACallInst(instr) is not None:
                    callee = LLVMGetCalledValue(instr)
                    callee_name = _llvm_value_name_as_str(callee)
                    # expand alias
                    if LLVMIsAGlobalAlias(callee):
                        callee = function_aliases[callee_name]
                        callee_name = _llvm_value_name_as_str(callee)
                    # only descend if the function is not inline assembly
                    if LLVMIsAInlineAsm(callee) is None:
                        identify_callees_(callee_name, callee)
                instr = LLVMGetNextInstruction(instr)
            bb = LLVMGetNextBasicBlock(bb)

    # loop over all functions
    for fn, name in _iter_functions(mod):
        name = _llvm_value_name_as_str(fn)
        call_conv = LLVMGetFunctionCallConv(fn)
        if call_conv == LLVMCallConv.LLVMAMDGPUKERNELCallConv:
            visibility = LLVMGetVisibility(fn)
            assert visibility == LLVMVisibility.LLVMProtectedVisibility
            identify_callees_(name, fn)
        elif LLVMIsDeclaration(fn) == 0:
            LLVMSetVisibility(fn, LLVMVisibility.LLVMHiddenVisibility)
            LLVMSetLinkage(fn, LLVMLinkage.LLVMPrivateLinkage)
    return used_functions


# def _module_remove_unused_functions(
#     mod, used_functions, function_aliases
# ):  # type: (LLVMOpaqueModule, set[str], dict[str,str]) -> None
#     """Remove unused functions from an LLVM module.

#     Note:
#         This causes segmentation fault.
#         Need to investigate more carefully how this
#         can be avoided. We use _module_remove_unused_functions_v2 instead.
#     """
#     for fn, name in _iter_functions(mod):
#         if name not in used_functions:
#             LLVMDeleteFunction(fn)

#     for alias, alias_name in _iter_global_function_aliases(mod):
#         if alias_name in function_aliases:
#             fn_name = _llvm_value_name_as_str(function_aliases[alias_name])
#             if fn_name not in used_functions:
#                 LLVMDeleteGlobal(alias)


def _tokenize_function_def_or_decl_header_line(line):
    """Tokenize a function declaration/definition line.

    Splits at whitespace as well as at '(', ')', and ','.
    Only keeps the latter three.

    Note:
        Since `p.split(line)` produces some '' and None entries, we
        need to remove those via the list comprehension that
        is part of the return statement.
    """
    p = re.compile(r"\s+|([(),])")
    return [tk for tk in p.split(line) if tk]


def _module_remove_unused_functions_v2(
    mod, used_functions, function_aliases
):  # type: (LLVMOpaqueModule, set[str], dict[str,str]) -> str
    ir_buf = _to_ir(mod)
    used_attributes = set()

    recording = True
    recorded_lines = []
    for line in ir_buf.decode().splitlines(keepends=True):
        if line.startswith("define ") or line.startswith("declare "):
            # example: define internal range(i64 0, 1024) i64 @__ockl_get_local_id(i32 noundef %0) local_unnamed_addr #22 {
            tokens = _tokenize_function_def_or_decl_header_line(line)

            name_tk = next((tk for tk in tokens if tk.startswith("@")), None)
            assert name_tk
            if name_tk[1:] in used_functions:
                recorded_lines.append(line)
                recording = True

                attrib_tk = next(
                    (tk for tk in tokens if tk.startswith("#")), None
                )
                if attrib_tk:
                    used_attributes.add(int(attrib_tk[1:]))
            else:
                # Turn recording off when uncalled function is encountered
                recording = False
                if recorded_lines[-1].startswith("; Function Attrs:"):
                    recorded_lines.pop()
                    if recorded_lines[-1].strip() == "":
                        recorded_lines.pop()
        elif line.startswith("}"):
            if recording:
                recorded_lines.append(line)
            else:
                # Turn it back on when "}" after is encountered
                # but do not record this line
                recording = True
        elif line.startswith("@") and "alias" in line:
            # example: @__ocml_cvtrtz_f32_u32 = internal alias float (i32), ptr @__ocml_cvtrtn_f32_u32
            alias_name = line[1:].split("=")[0].strip()
            fn_name = bytes(function_aliases[alias_name]).decode()
            if fn_name in used_functions:
                recorded_lines.append(line)
            else:
                pass
        elif line.startswith("attributes #"):
            attrib_tk = next(
                (tk for tk in line.split(" ") if tk.startswith("#")), None
            )
            if attrib_tk:
                if int(attrib_tk[1:]) in used_attributes:
                    recorded_lines.append(line)
        else:
            if recording:
                recorded_lines.append(line)
    return "".join(recorded_lines)


def clean_up_kernel_module(
    mod, mod_len: int = -1, to_bc: bool = False
):  # noqa: C901
    """

    Deletes any function that is not called (directly or via alias) by a protected function or is a protected function itself.
    Changes the visibility of all non-protected functions to 'hidden' and
    their linkage to 'private' so that they will not be part of
    any

    Note:
        We assume here that AMD `amdgpu_kernel` functions have protected
        visibility and thus use this visibilty as indicator for identifying kernels.

    Args:
        mod (UTF-8 `str`, or implementor of the Python buffer protocol such as
            `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
            Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
        mod_len (`int`, optional):
            Length of the LLVM IR buffer. Callers can specify numbers smaller than 1
            or ``None`` to indicate that the buffer length should be derived via ``len(mod)``.
            Defaults to ``-1``. Not used at all if ``mod`` is an instance of
            `rocm.llvm.c.types.LLVMOpaqueModule`.
        to_bc (`bool`, optional):
            Return LLVM bitcode (True) or LLVM IR (False).
            Defaults to True.

    Returns:


    TODO:
        We make implicit assumptions on the LLVM IR format here (function head in one line, attributes in one line, ...) that need
        to be documented.
    """

    def clean_module_(mod, to_bc):  # type: (LLVMOpaqueModule, bool) -> bytes
        """
        We currently do not remove superfluous attributes as
        we the attributes might actually be used by called functions.
        """
        function_aliases = _get_function_aliases(mod)
        used_functions = (
            _get_functions_called_by_amdgpu_kernel_hide_device_functions(
                mod, function_aliases
            )
        )

        result = _module_remove_unused_functions_v2(
            mod, used_functions, function_aliases
        )

        # print(result)

        if to_bc:
            res = to_bc_from_ir(result.encode())
            return res
        else:
            return result.encode()

    if isinstance(mod, LLVMOpaqueModule):
        return clean_module_(mod, to_bc)
    else:
        context = LLVMContextCreate()
        (llvm_mod,) = _get_module_in_context(context, mod, mod_len)
        result = clean_module_(llvm_mod, to_bc)
        # LLVMDisposeModule(mod)  # NOTE: module seems to be owned by context
        LLVMContextDispose(context)
        return result


def is_human_readable_clang_offload_bundle(filecontent: str):
    """Checks if a file is a bundle of (human-readable) LLVM IR.

    Note:
        Human-readable clang offload bundles contain
        strings such as "; __CLANG_OFFLOAD_BUNDLE____START__ <target-id>"
        and "; __CLANG_OFFLOAD_BUNDLE____END__ <target-id>".
    Note:
        Clang offload bundles that contain bitcode
        contain strings such as: `__CLANG_OFFLOAD_BUNDLE__<target-id>`
    """
    try:
        if isinstance(filecontent, bytes):
            filecontent = filecontent.decode("utf-8")
        return "; __CLANG_OFFLOAD_BUNDLE____END__" in filecontent
    # TODO: specialize error type
    except Exception:
        return False


def amdgpu_target_id(amdgpu_arch: str):
    """Returns a target ID for the AMD GPU arch.

    The resulting string can be used as key
    for results of `split_human_readable_clang_offload_bundle`.
    """
    return f"hip-amdgcn-amd-amdhsa--{amdgpu_arch}"


def split_human_readable_clang_offload_bundle(bundle):
    """Splits a human-readable LLVM IR bundle into its parts.

    Example:

        ```llvm
        ; __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
        ; ...
        ; __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
        ; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
        ; ...
        ; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
        ```

        will reproduce a dictionary with the two keys 'hip-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-'
        and 'host-x86_64-unknown-linux-gnu-'.

    Note:
        Does not check if "; __CLANG_OFFLOAD_BUNDLE____END__ " is followed by the correct label.

    Returns:
        `dict`:
            A `dict` that holds an IR module per detected target ID.
    """
    result = {}
    if isinstance(bundle, bytes):
        bundle = bundle.decode("utf-8")
    else:
        RuntimeError("expected `str` or `bytes`")
    assert isinstance(bundle, str)

    p_begin = "; __CLANG_OFFLOAD_BUNDLE____START__ "
    p_end = "; __CLANG_OFFLOAD_BUNDLE____END__ "

    cursor: int = 0
    while True:
        begin: int = bundle.find(
            p_begin, cursor
        )  # note: returns -1 on failure
        if begin < 0:
            break
        else:
            next_newline: int = bundle.find("\n", begin)
            target_id: str = bundle[
                begin + len(p_begin) : next_newline
            ]  # noqa: E203
            begin = next_newline + 1  # move at begin of next line
            end: int = bundle.find(p_end, begin)  # note: returns -1 on failure
            if end == -1:
                raise RuntimeError(
                    "no matching __CLANG_OFFLOAD_BUNDLE____END__ found"
                )
            else:
                result[target_id] = bundle[
                    begin:end
                ]  # note: exclusive upper bound
                cursor = end
    return result
