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
    if ir_len == None or ir_len < 1:
        ir_len = len(ir)

    buf = LLVMCreateMemoryBufferWithMemoryRange(
        ir,
        ir_len,
        b"llvm-ir-buffer",
        0,
    )
    # (status, mod, message)
    # TODO HIP check memory, mod seems to take ownership of the buffer
    return LLVMParseIRInContext(LLVMGetGlobalContext(), buf)


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
            A `tuple` ``(mod, )`` that contains (in that order):

            * mod - The loaded LLVM module.
    Note:
        Return list might need to be extended,
        hence the tuple result.
    See:
        _get_module_dispose_all
    """
    (status, mod, err_cstr) = _parse_llvm_ir(ir, ir_len)
    if status > 0:  # failure
        errmsg = err_cstr.decode("utf-8")
        LLVMDisposeMessage(err_cstr)
        if mod:
            LLVMDisposeModule(mod)
        # LLVMDisposeMemoryBuffer(ir_buf) mod seems to take ownership of the buffer # TODO HIP check memory
        raise ValueError(
            "input 'buf' seems to be neither valid LLVM bitcode nor LLVM assembly.\n\n"
            f"Reason: {errmsg}"
        )
    else:
        return (mod,)


def _get_module_dispose_all(mod):
    """Clean up the results of `_get_module`.

    Args:
        mod: A module.

    Note:
        Arg list might need to be extended,
        hence the name `_get_module_dispose_all`.
    See:
        _get_module_dispose_all
    """
    LLVMDisposeModule(mod)


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
    """LLVM bitcode as humand-readable LLVM assembly.

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
    """Human-readable LLVM assembly or LLVM bitcode as LLVM bitcode.

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


def to_ir_fast(mod, mod_len: int = -1):
    """Convert human-readable LLVM IR or LLVM bitcode to human-readable LLVM IR.

    Fast version of `to_ir` that does not return a copy
    if the input is already human-readable LLVM assembly.
    This routines assumes that the input is human-readable
    IR if the first two bytes of the input 'mod' are not the
    ASCII chars "BC".

    Returns:
        `bytes` or `str`:
            Returns the result as
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
        gm_res = _get_module(mod, mod_len)
        result = _to_bc(mod=gm_res[0])
        _get_module_dispose_all(*gm_res)
        return result


def to_bc_fast(mod, mod_len: int = -1):
    """Convert human-readable LLVM IR or LLVM bitcode to LLVM bitcode.

    Fast version of `to_bc` that does not return a copy
    if the input is already human-readable LLVM assembly.
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
            cloned_modules.append((LLVMCloneModule(entry), None))
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
    except:
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
    target_id = None
    for line in bundle.splitlines(keepends=True):
        if line.lstrip().startswith(p_begin):
            assert target_id == None
            target_id = line.replace(p_begin, "").strip()
            result[target_id] = ""
        elif line.lstrip().startswith(p_end):
            assert target_id != None
            target_id = None
        else:
            if target_id != None:
                result[target_id] += line
    return result


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
        begin: int = bundle.find(p_begin, cursor)  # note: returns -1 on failure
        if begin < 0:
            break
        else:
            next_newline: int = bundle.find("\n", begin)
            target_id: str = bundle[begin + len(p_begin) : next_newline]
            begin = next_newline + 1  # move at begin of next line
            end: int = bundle.find(p_end, begin)  # note: returns -1 on failure
            if end == -1:
                raise RuntimeError("no matching __CLANG_OFFLOAD_BUNDLE____END__ found")
            else:
                result[target_id] = bundle[begin:end]  # note: exclusive upper bound
                cursor = end
    return result
