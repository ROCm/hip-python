# Copyright (c) 2012, Anaconda, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

import re
from functools import cached_property
import llvmlite.binding as ll
from llvmlite import ir

from numba.core import typing, types, debuginfo, itanium_mangler, cgutils
from numba.core.dispatcher import Dispatcher
from numba.core.base import BaseContext
from numba.core.callconv import MinimalCallConv
from numba.core.typing import cmathdecl
from numba.core import datamodel

# from .hipdrv import nvvm
from . import amdgcn
from numba import hip
from numba.hip import codegen  # , nvvmutils, ufuncs
from numba.hip.typing_lowering import stubs, ufuncs
from numba.hip.typing_lowering.models import hip_data_manager
from numba.hip import typing_lowering

# -----------------------------------------------------------------------------
# Typing

# TODO(HIP/AMD): Must be reloaded in case C++ extensions (experimental Numba-HIP-only feature under development) are added.
module_hip_attributes = {}
module_hip_attributes.update(typing_lowering.hipdevicelib.thestubs)
module_hip_attributes.update(typing_lowering.hip.thestubs)

stubs.resolve_attributes(
    typing_lowering.registries.typing_registry, hip, module_hip_attributes
)


class HIPTypingContext(typing.BaseContext):
    def load_additional_registries(self):
        from . import typing_lowering
        from numba.core.typing import enumdecl, cffi_utils

        self.install_registry(cffi_utils.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(enumdecl.registry)

        self.install_registry(typing_lowering.typing_registry)
        # self.install_registry(cudadecl.registry)
        # self.install_registry(cudamath.registry)
        # self.install_registry(libdevicedecl.registry)
        # self.install_registry(vector_types.typing_registry)

    def resolve_value_type(self, val):
        # treat other dispatcher object as another device function
        from numba.hip.dispatcher import HIPDispatcher

        if isinstance(val, Dispatcher) and not isinstance(val, HIPDispatcher):
            try:
                # use cached device function
                val = val.__dispatcher
            except AttributeError:
                if not val._can_compile:
                    raise ValueError(
                        "using cpu function on device "
                        "but its compilation is disabled"
                    )
                targetoptions = val.targetoptions.copy()
                targetoptions["device"] = True
                targetoptions["debug"] = targetoptions.get("debug", False)
                targetoptions["opt"] = targetoptions.get("opt", True)
                disp = HIPDispatcher(val.py_func, targetoptions)
                # cache the device function for future use and to avoid
                # duplicated copy of the same function.
                val.__dispatcher = disp
                val = disp

        # continue with parent logic
        return super(HIPTypingContext, self).resolve_value_type(val)


# -----------------------------------------------------------------------------
# Implementation


VALID_CHARS = re.compile(r"[^a-z0-9]", re.I)


class HIPTargetContext(BaseContext):
    implement_powi_as_math_call = True
    strict_alignment = True

    def __init__(self, typingctx, target="hip"):
        super().__init__(typingctx, target)
        self.data_model_manager = hip_data_manager.chain(datamodel.default_manager)

    @property
    def DIBuilder(self):
        return debuginfo.DIBuilder

    @property
    def enable_boundscheck(self):
        # Unconditionally disabled
        return False

    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    def init(self):
        self._internal_codegen = codegen.JITHIPCodegen("numba.hip.jit")
        self._target_data = None

    def load_additional_registries(self):
        # side effect of import needed for numba.cpython.*, the builtins
        # registry is updated at import time.
        # !! imports have side effects !!
        from numba.cpython import numbers, tupleobj, slicing  # noqa: F401
        from numba.cpython import rangeobj, iterators, enumimpl  # noqa: F401
        from numba.cpython import unicode, charseq  # noqa: F401
        from numba.cpython import cmathimpl
        from numba.misc import cffiimpl
        from numba.np import arrayobj  # noqa: F401
        from numba.np import npdatetime  # noqa: F401

        # from . import (
        #     cudaimpl, printimpl, libdeviceimpl, mathimpl, vector_types
        # )
        from . import typing_lowering

        # fix for #8940
        from numba.np.unsafe import ndarray  # noqa F401

        self.install_registry(cffiimpl.registry)
        self.install_registry(cmathimpl.registry)

        self.install_registry(typing_lowering.impl_registry)
        # self.install_registry(hipimpl.registry)
        # self.install_registry(printimpl.registry)
        # self.install_registry(libdeviceimpl.registry)
        # self.install_registry(mathimpl.registry)
        # self.install_registry(vector_types.impl_registry)

    def codegen(self):
        return self._internal_codegen

    @property
    def target_data(self):
        if self._target_data is None:
            self._target_data = ll.create_target_data(amdgcn.DATA_LAYOUT)
        return self._target_data

    @cached_property
    def nonconst_module_attrs(self):
        """
        Some HIP intrinsics are at the module level, but cannot be treated as
        constants, because they are loaded from a special register in the PTX.
        These include threadIdx, blockDim, etc.
        """
        from numba import hip

        nonconsts = (  # TODO(HIP/AMD) check this
            "threadIdx",
            "blockDim",
            "blockIdx",
            "gridDim",
            "laneid",
            "warpsize",
        )
        nonconsts_with_mod = tuple([(types.Module(hip), nc) for nc in nonconsts])
        return nonconsts_with_mod

    @cached_property
    def call_conv(self):
        return HIPCallConv(self)

    def mangler(self, name, argtypes, *, abi_tags=(), uid=None):
        return itanium_mangler.mangle(name, argtypes, abi_tags=abi_tags, uid=uid)

    def prepare_hip_kernel(
        self,
        codelib,
        fndesc,
        debug,
        lineinfo,
        options,
        filename,
        linenum,
        max_registers=None,
        name: str = None,
    ):
        """
        Adapt a code library ``codelib`` with the numba compiled HIP kernel
        with name ``fname`` and arguments ``argtypes`` for NVVM.
        A new library is created with a wrapper function that can be used as
        the kernel entry point for the given kernel.

        Returns the new code library and the wrapper function.

        Args:
            codelib:
                The CodeLibrary containing the device function to wrap in a kernel call.
            fndesc:
                The FunctionDescriptor of the source function.
            debug:
                Whether to compile with debug.
            lineinfo:
                Whether to emit line info.
            options:
                Dict of options used when compiling the new library.
            filename:
                The source filename that the function is contained in.
            linenum:
                The source line that the function is on.
            max_registers: The max_registers argument for the code library.

        Returns:

        """
        kernel_name = itanium_mangler.prepend_namespace(
            fndesc.llvm_func_name,
            ns="hippy",
        )
        library: codegen.HIPCodeLibrary = self.codegen().create_library(
            f"{codelib.name}_kernel_",
            entry_name=kernel_name,
            options=options,
            max_registers=max_registers,
            device=False,
        )
        if name:
            library.change_entry_name(name)
            kernel_name = library._entry_name
        library.add_linking_library(codelib)
        wrapper = self.generate_kernel_wrapper(
            library, fndesc, kernel_name, debug, lineinfo, filename, linenum
        )
        return library, wrapper

    def generate_kernel_wrapper(
        self, library, fndesc, kernel_name, debug, lineinfo, filename, linenum
    ):
        """
        Generate the kernel wrapper in the given ``library``.
        The function being wrapped is described by ``fndesc``.
        The wrapper function is returned.
        """

        argtypes = fndesc.argtypes
        arginfo = self.get_arg_packer(argtypes)
        argtys = list(arginfo.argument_types)
        wrapfnty = ir.FunctionType(ir.VoidType(), argtys)
        wrapper_module = self.create_module("hip.kernel.wrapper")
        fnty = ir.FunctionType(
            ir.IntType(32), [self.call_conv.get_return_type(types.pyobject)] + argtys
        )
        func = ir.Function(wrapper_module, fnty, fndesc.llvm_func_name)

        prefixed = itanium_mangler.prepend_namespace(func.name, ns="hippy")
        wrapfn = ir.Function(wrapper_module, wrapfnty, prefixed)
        builder = ir.IRBuilder(wrapfn.append_basic_block(""))

        if debug or lineinfo:
            pass
            # TODO enable debugging
            # directives_only = lineinfo and not debug
            # debuginfo = self.DIBuilder(
            #     module=wrapper_module,
            #     filepath=filename,
            #     cgctx=self,
            #     directives_only=directives_only,
            # )
            # debuginfo.mark_subprogram(
            #     wrapfn,
            #     kernel_name,
            #     fndesc.args,
            #     argtypes,
            #     linenum,
            # )
            # debuginfo.mark_location(builder, linenum)

        # Define error handling variable
        def define_error_gv(postfix):
            name = wrapfn.name + postfix
            gv = cgutils.add_global_variable(wrapper_module, ir.IntType(32), name)
            gv.initializer = ir.Constant(gv.type.pointee, None)
            return gv

        gv_exc = define_error_gv("__errcode__")
        gv_tid = []
        gv_ctaid = []
        for i in "xyz":
            gv_tid.append(define_error_gv("__tid%s__" % i))
            gv_ctaid.append(define_error_gv("__ctaid%s__" % i))

        callargs = arginfo.from_arguments(builder, wrapfn.args)
        status, _ = self.call_conv.call_function(
            builder, func, types.void, argtypes, callargs
        )

        if debug:
            raise NotImplementedError()
            # TODO support debug mode
            # Check error status
            # with cgutils.if_likely(builder, status.is_ok):
            #     builder.ret_void()

            # with builder.if_then(builder.not_(status.is_python_exc)):
            #     # User exception raised
            #     old = ir.Constant(gv_exc.type.pointee, None)

            #     # Use atomic cmpxchg to prevent rewriting the error status
            #     # Only the first error is recorded

            #     xchg = builder.cmpxchg(
            #         gv_exc, old, status.code, "monotonic", "monotonic"
            #     )
            #     changed = builder.extract_value(xchg, 1)

            #     # If the xchange is successful, save the thread ID.
            #     sreg = nvvmutils.SRegBuilder(builder)
            #     with builder.if_then(changed):
            #         for (
            #             dim,
            #             ptr,
            #         ) in zip("xyz", gv_tid):
            #             val = sreg.tid(dim)
            #             builder.store(val, ptr)

            #         for (
            #             dim,
            #             ptr,
            #         ) in zip("xyz", gv_ctaid):
            #             val = sreg.ctaid(dim)
            #             builder.store(val, ptr)

        builder.ret_void()

        # nvvm.set_hip_kernel(wrapfn) # TODO
        library.add_ir_module(wrapper_module)
        if debug or lineinfo:
            debuginfo.finalize()
        library.finalize()
        wrapfn = library.get_function(wrapfn.name)
        return wrapfn

    def make_constant_array(self, builder, aryty, arr):
        """
        Unlike the parent version.  This returns a a pointer in the constant
        addrspace.
        """

        lmod = builder.module

        constvals = [
            self.get_constant(types.byte, i) for i in iter(arr.tobytes(order="A"))
        ]
        constaryty = ir.ArrayType(ir.IntType(8), len(constvals))
        constary = ir.Constant(constaryty, constvals)

        addrspace = amdgcn.ADDRSPACE_CONSTANT
        gv = cgutils.add_global_variable(
            lmod, constary.type, "_hippy_cmem", addrspace=addrspace
        )
        gv.linkage = "internal"
        gv.global_constant = True
        gv.initializer = constary

        # Preserve the underlying alignment
        lldtype = self.get_data_type(aryty.dtype)
        align = self.get_abi_sizeof(lldtype)
        gv.align = 2 ** (align - 1).bit_length()

        # Convert to generic address-space
        ptrty = ir.PointerType(ir.IntType(8))
        genptr = builder.addrspacecast(gv, ptrty, "generic")

        # Create array object
        ary = self.make_array(aryty)(self, builder)
        kshape = [self.get_constant(types.intp, s) for s in arr.shape]
        kstrides = [self.get_constant(types.intp, s) for s in arr.strides]
        self.populate_array(
            ary,
            data=builder.bitcast(genptr, ary.data.type),
            shape=kshape,
            strides=kstrides,
            itemsize=ary.itemsize,
            parent=ary.parent,
            meminfo=None,
        )

        return ary._getvalue()

    def insert_const_string(self, mod, string):
        """
        Unlike the parent version.  This returns a a pointer in the constant
        addrspace.
        """
        text = cgutils.make_bytearray(string.encode("utf-8") + b"\x00")
        name = "$".join(["__conststring__", itanium_mangler.mangle_identifier(string)])
        # Try to reuse existing global
        global_var = mod.globals.get(name)
        if global_var is None:
            # Not defined yet
            global_var = cgutils.add_global_variable(
                mod, text.type, name, addrspace=amdgcn.ADDRSPACE_CONSTANT
            )
            global_var.linkage = "internal"
            global_var.global_constant = True
            global_var.initializer = text

        # Cast to a i8* pointer
        charty = global_var.type.pointee.element
        return global_var.bitcast(charty.as_pointer(amdgcn.ADDRSPACE_CONSTANT))

    def insert_string_const_addrspace(self, builder, string):
        """
        Insert a constant string in the constant addresspace and return a
        generic i8 pointer to the data.

        This function attempts to deduplicate.
        """
        lmod = builder.module
        gv = self.insert_const_string(lmod, string)
        charptrty = ir.PointerType(ir.IntType(8))
        return builder.addrspacecast(gv, charptrty, "generic")

    def get_ufunc_info(self, ufunc_key):
        return ufuncs.get_ufunc_info(ufunc_key)


class HIPCallConv(MinimalCallConv):
    pass
