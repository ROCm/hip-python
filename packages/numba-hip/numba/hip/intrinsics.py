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

"""

Note:
    In contrast to Numba CUDA, the following implementations are already provided 
    via hipdevicelib (vs. Numba CUDA libdevice):

    * ``syncthreads``
    * ``syncthreads_count``
    * ``syncthreads_and``
    * ``syncthreads_or``
"""

from llvmlite import ir

from numba import hip, types
from numba.core import cgutils
from numba.core.errors import RequireLiteralValue
from numba.core.typing import signature
from numba.core.extending import overload_attribute

#: from numba.cuda import nvvmutils # TODO: HIP/AMD: not supported
from numba.hip.extending import intrinsic

from numba.hip.hipdevicelib import (
    global_id as _global_id,  # these stubs are created at runtime,
    gridsize as _gridsize,  # see numba.hip.hipdevicelib.HIPDeviceLib._create_extensions,
    warpsize as _warpsize_fun,
)


def _call_first(stub, *args):
    """Runs the first call generator registered with the stub."""
    return stub._call_generators_[0](*args)


# -------------------------------------------------------------------------------
# Grid functions


def _type_grid_function(ndim):
    val = ndim.literal_value
    if val == 1:
        restype = types.int32
    elif val in (2, 3):
        restype = types.UniTuple(types.int32, val)
    else:
        raise ValueError("argument can only be 1, 2, 3")

    return signature(restype, types.int32)


@intrinsic
def grid(typingctx, ndim):
    """grid(ndim)

    Return the absolute position of the current thread in the entire grid of
    blocks.  *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """
    if not isinstance(ndim, types.IntegerLiteral):
        raise RequireLiteralValue(ndim)

    sig = _type_grid_function(ndim)

    def codegen(context, builder, sig, args):
        restype = sig.return_type
        if restype == types.int32:
            # return nvvmutils.get_global_id(builder, dim=1)
            return _call_first(_global_id.x)
        elif isinstance(restype, types.UniTuple):
            # ids = nvvmutils.get_global_id(builder, dim=restype.count)
            ids = [
                _call_first(stub)
                for stub in (_global_id.x, _global_id.y, _global_id.z)[: restype.count]
            ]
            return cgutils.pack_array(builder, ids)

    return sig, codegen


@intrinsic
def gridsize(typingctx, ndim):
    """gridsize(ndim)

    Return the absolute size (or shape) in threads of the entire grid of
    blocks. *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.blockDim.x * cuda.gridDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """
    if not isinstance(ndim, types.IntegerLiteral):
        raise RequireLiteralValue(ndim)

    sig = _type_grid_function(ndim)

    def codegen(context, builder, sig, args):
        restype = sig.return_type
        if restype == types.int32:
            # return nvvmutils.get_global_id(builder, dim=1)
            return _call_first(_gridsize.x)
        elif isinstance(restype, types.UniTuple):
            # ids = nvvmutils.get_global_id(builder, dim=restype.count)
            ids = [
                _call_first(stub)
                for stub in (_gridsize.x, _gridsize.y, _gridsize.z)[: restype.count]
            ]
            return cgutils.pack_array(builder, ids)

    return sig, codegen


@intrinsic
def _warpsize(typingctx):
    sig = signature(types.int32)

    def codegen(context, builder, sig, args):
        return _call_first(_warpsize_fun)

    return sig, codegen


@overload_attribute(types.Module(hip), "warpsize", target="hip")
def hip_warpsize(mod):
    """
    The size of a warp. All architectures implemented to date have a warp size
    of 64.
    """

    def get(mod):
        return _warpsize()

    return get
