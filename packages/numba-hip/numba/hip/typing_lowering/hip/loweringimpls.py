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

"""Lowering implementations for HIP types such as dim3 and arrays in GPU address spaces.

Note:
    This module is derived from `numba/HIP/HIPimpl.py`.
TODO:
    * Support cooperative groups.
    * Support fp16/half precision floats.
    * Support atomics for arrays types.
    * Support other math functions such as round/radians -> Best via hipdevicelib C++ extension.

Attributes:
    typing_registry (`numba.core.typing.templates.Registry`):
        A registry of typing declarations. The registry stores such declarations
        for functions, attributes and globals.
"""

from functools import reduce
import operator
import math

from llvmlite import ir
import llvmlite.binding as ll

from numba.core.imputils import Registry, lower_cast
from numba.core.typing.npydecl import parse_dtype
from numba.core.datamodel import models
from numba.core import types, cgutils
from numba.np import ufunc_db
from numba.np.npyimpl import register_ufuncs
import numba.hip.amdgcn as amdgcn
from numba.hip.typing_lowering import stubs
# from numba import HIP
# from numba.HIP import nvvmutils, stubs, errors
from numba.hip.typing_lowering.types import dim3, HIPDispatcher

impl_registry = Registry()
lower = impl_registry.lower
lower_attr = impl_registry.lower_getattr
lower_constant = impl_registry.lower_constant

@lower_attr(dim3, 'x')
def dim3_x(context, builder, sig, args):
    return builder.extract_value(args, 0)


@lower_attr(dim3, 'y')
def dim3_y(context, builder, sig, args):
    return builder.extract_value(args, 1)


@lower_attr(dim3, 'z')
def dim3_z(context, builder, sig, args):
    return builder.extract_value(args, 2)

# -----------------------------------------------------------------------------

@lower(stubs.const.array_like, types.Array)
def hip_const_array_like(context, builder, sig, args):
    # This is a no-op because HIPTargetContext.make_constant_array already
    # created the constant array.
    return args[0]


_unique_smem_id = 0


def _get_unique_smem_id(name):
    """Due to bug with NVVM invalid internalizing of shared memory in the
    PTX output.  We can't mark shared memory to be internal. We have to
    ensure unique name is generated for shared memory symbol.
    """
    global _unique_smem_id
    _unique_smem_id += 1
    return "{0}_{1}".format(name, _unique_smem_id)


@lower(stubs.shared.array, types.IntegerLiteral, types.Any)
def hip_shared_array_integer(context, builder, sig, args):
    length = sig.args[0].literal_value
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=(length,), dtype=dtype,
                          symbol_name=_get_unique_smem_id('_HIPpy_smem'),
                          addrspace=amdgcn.ADDRSPACE_SHARED,
                          can_dynsized=True)


@lower(stubs.shared.array, types.Tuple, types.Any)
@lower(stubs.shared.array, types.UniTuple, types.Any)
def hip_shared_array_tuple(context, builder, sig, args):
    shape = [ s.literal_value for s in sig.args[0] ]
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=shape, dtype=dtype,
                          symbol_name=_get_unique_smem_id('_HIPpy_smem'),
                          addrspace=amdgcn.ADDRSPACE_SHARED,
                          can_dynsized=True)


@lower(stubs.local.array, types.IntegerLiteral, types.Any)
def hip_local_array_integer(context, builder, sig, args):
    length = sig.args[0].literal_value
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=(length,), dtype=dtype,
                          symbol_name='_HIPpy_lmem',
                          addrspace=amdgcn.ADDRSPACE_LOCAL,
                          can_dynsized=False)


@lower(stubs.local.array, types.Tuple, types.Any)
@lower(stubs.local.array, types.UniTuple, types.Any)
def ptx_lmem_alloc_array(context, builder, sig, args):
    shape = [ s.literal_value for s in sig.args[0] ]
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=shape, dtype=dtype,
                          symbol_name='_HIPpy_lmem',
                          addrspace=amdgcn.ADDRSPACE_LOCAL,
                          can_dynsized=False)

def _generic_array(context, builder, shape, dtype, symbol_name, addrspace,
                   can_dynsized=False):
    elemcount = reduce(operator.mul, shape, 1)

    # Check for valid shape for this type of allocation.
    # Only 1d arrays can be dynamic.
    dynamic_smem = elemcount <= 0 and can_dynsized and len(shape) == 1
    if elemcount <= 0 and not dynamic_smem:
        raise ValueError("array length <= 0")

    # Check that we support the requested dtype
    data_model = context.data_model_manager[dtype]
    other_supported_type = (
        isinstance(dtype, (types.Record, types.Boolean))
        or isinstance(data_model, models.StructModel)
        or dtype == types.float16
    )
    if dtype not in types.number_domain and not other_supported_type:
        raise TypeError("unsupported type: %s" % dtype)

    lldtype = context.get_data_type(dtype)
    laryty = ir.ArrayType(lldtype, elemcount)

    if addrspace == amdgcn.ADDRSPACE_LOCAL:
        # Special case local address space allocation to use alloca
        # NVVM is smart enough to only use local memory if no register is
        # available
        dataptr = cgutils.alloca_once(builder, laryty, name=symbol_name)
    else:
        lmod = builder.module

        # Create global variable in the requested address space
        gvmem = cgutils.add_global_variable(lmod, laryty, symbol_name,
                                            addrspace)
        # Specify alignment to avoid misalignment bug
        align = context.get_abi_sizeof(lldtype)
        # Alignment is required to be a power of 2 for shared memory. If it is
        # not a power of 2 (e.g. for a Record array) then round up accordingly.
        gvmem.align = 1 << (align - 1 ).bit_length()

        if dynamic_smem:
            gvmem.linkage = 'external'
        else:
            ## Comment out the following line to workaround a NVVM bug
            ## which generates a invalid symbol name when the linkage
            ## is internal and in some situation.
            ## See _get_unique_smem_id()
            # gvmem.linkage = lc.LINKAGE_INTERNAL

            gvmem.initializer = ir.Constant(laryty, ir.Undefined)

        # Convert to generic address-space
        dataptr = builder.addrspacecast(gvmem, ir.PointerType(ir.IntType(8)),
                                        'generic')

    targetdata = ll.create_target_data(amdgcn.DATA_LAYOUT) # TODO numba.hip: Potentially needs to be configured by compiler before lowering
    lldtype = context.get_data_type(dtype)
    itemsize = lldtype.get_abi_size(targetdata)

    # Compute strides
    laststride = itemsize
    rstrides = []
    for i, lastsize in enumerate(reversed(shape)):
        rstrides.append(laststride)
        laststride *= lastsize
    strides = [s for s in reversed(rstrides)]
    kstrides = [context.get_constant(types.intp, s) for s in strides]

    # Compute shape
    if dynamic_smem:
        raise NotImplementedError("numba.hip: no support for dynamic shared memory implemented yet")
        # # Compute the shape based on the dynamic shared memory configuration.
        # # Unfortunately NVVM does not provide an intrinsic for the
        # # %dynamic_smem_size register, so we must read it using inline
        # # assembly.
        # get_dynshared_size = ir.InlineAsm(ir.FunctionType(ir.IntType(32), []),
        #                                   "mov.u32 $0, %dynamic_smem_size;",
        #                                   '=r', side_effect=True)
        # dynsmem_size = builder.zext(builder.call(get_dynshared_size, []),
        #                             ir.IntType(64))
        # # Only 1-D dynamic shared memory is supported so the following is a
        # # sufficient construction of the shape
        # kitemsize = context.get_constant(types.intp, itemsize)
        # kshape = [builder.udiv(dynsmem_size, kitemsize)]
    else:
        kshape = [context.get_constant(types.intp, s) for s in shape]

    # Create array object
    ndim = len(shape)
    aryty = types.Array(dtype=dtype, ndim=ndim, layout='C')
    ary = context.make_array(aryty)(context, builder)

    context.populate_array(ary,
                           data=builder.bitcast(dataptr, ary.data.type),
                           shape=kshape,
                           strides=kstrides,
                           itemsize=context.get_constant(types.intp, itemsize),
                           meminfo=None)
    return ary._getvalue()


@lower_constant(HIPDispatcher)
def hip_dispatcher_const(context, builder, ty, pyval):
    return context.get_dummy_value()