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

"""Registers typing declarations and implementations for Python math function objects.

Attributes:
    typing_registry (`numba.core.typing.templates.Registry`):
        A registry of typing declarations. The registry stores such declarations
        for functions, attributes and globals.
    impl_registry (`numba.core.imputils.Registry`):
        A registry of function and attribute implementations.
"""

import math

from numba.core import types, typing

from numba.hip.typing_lowering.registries import (
    typing_registry,
    impl_registry,
)
from numba.hip.typing_lowering import hipdevicelib, stubs as numba_hip_stubs

thestubs = {}
for _name, _mathobj in vars(math).items():
    if callable(_mathobj):  # only consider functions
        _stub = hipdevicelib.thestubs.get(_name, None)
        if _stub != None:
            # register hipdevicelib typing template and implementation/lowering procedures
            # for this math object
            # NOTE:
            #     We still collect the stubs to inform the `math` module
            #     attribute resolution in `numba/hip/target.py`.
            thestubs[_name] = numba_hip_stubs.Stub.from_other(
                _stub,
                _mathobj,
                f"NUMBA_HIP_MATH_STUB_{_mathobj.__name__}",
                template_prefix="NUMBA_HIP_MATH_TEMPLATE_",
                register=True,
                typing_registry=typing_registry,
                impl_registry=impl_registry,
            )

#
# HIP: Below code unrelated to above, just to support import `ufuncs` module in __init__.py file.
#

unarys = [
    ("ceil", "ceilf", math.ceil),
    ("floor", "floorf", math.floor),
    ("fabs", "fabsf", math.fabs),
    ("exp", "expf", math.exp),
    ("expm1", "expm1f", math.expm1),
    ("erf", "erff", math.erf),
    ("erfc", "erfcf", math.erfc),
    ("tgamma", "tgammaf", math.gamma),
    ("lgamma", "lgammaf", math.lgamma),
    ("sqrt", "sqrtf", math.sqrt),
    ("log", "logf", math.log),
    ("log2", "log2f", math.log2),
    ("log10", "log10f", math.log10),
    ("log1p", "log1pf", math.log1p),
    ("acosh", "acoshf", math.acosh),
    ("acos", "acosf", math.acos),
    ("cos", "cosf", math.cos),
    ("cosh", "coshf", math.cosh),
    ("asinh", "asinhf", math.asinh),
    ("asin", "asinf", math.asin),
    ("sin", "sinf", math.sin),
    ("sinh", "sinhf", math.sinh),
    ("atan", "atanf", math.atan),
    ("atanh", "atanhf", math.atanh),
    ("tan", "tanf", math.tan),
    ("trunc", "truncf", math.trunc),
]


def get_lower_unary_impl(key, ty, libfunc):
    def lower_unary_impl(context, builder, sig, args):
        actual_libfunc = libfunc
        # TODO fast-math
        # fast_replacement = None
        # if ty == types.float32 and context.fastmath:
        #     fast_replacement = unarys_fastmath.get(libfunc.__name__)

        # if fast_replacement is not None:
        #     actual_libfunc = getattr(libdevice, fast_replacement)

        libfunc_impl = context.get_function(actual_libfunc, typing.signature(ty, ty))
        return libfunc_impl(builder, args)

    return lower_unary_impl


def get_unary_impl_for_fn_and_ty(fn, ty):
    # tanh is a special case - because it is not registered like the other
    # unary implementations, it does not appear in the unarys list. However,
    # its implementation can be looked up by key like the other
    # implementations, so we add it to the list we search here.
    tanh_impls = ("tanh", "tanhf", math.tanh)
    for fname64, fname32, key in unarys + [tanh_impls]:
        if fn == key:
            if ty == types.types.float32:
                impl = hipdevicelib.thestubs.get(fname32)
            elif ty == types.types.float64:
                impl = hipdevicelib.thestubs.get(fname64)

            return get_lower_unary_impl(key, ty, impl)

    raise RuntimeError(f"Implementation of {fn} for {ty} not found")


binarys = [
    ("copysign", "copysignf", math.copysign),
    ("atan2", "atan2f", math.atan2),
    ("pow", "powf", math.pow),
    ("fmod", "fmodf", math.fmod),
    ("hypot", "hypotf", math.hypot),
    ("remainder", "remainderf", math.remainder),
]


def get_lower_binary_impl(key, ty, libfunc):
    def lower_binary_impl(context, builder, sig, args):
        actual_libfunc = libfunc
        # TODO fast-math
        # fast_replacement = None
        # if ty == types.types.float32 and context.fastmath:
        #     fast_replacement = binarys_fastmath.get(libfunc.__name__)

        # if fast_replacement is not None:
        #     actual_libfunc = getattr(libdevice, fast_replacement)

        libfunc_impl = context.get_function(
            actual_libfunc, typing.signature(ty, ty, ty)
        )
        return libfunc_impl(builder, args)

    return lower_binary_impl


def get_binary_impl_for_fn_and_ty(fn, ty):
    for fname64, fname32, key in binarys:
        if fn == key:
            if ty == types.types.float32:
                impl = hipdevicelib.thestubs.get(fname32)
            elif ty == types.types.float64:
                impl = hipdevicelib.thestubs.get(fname64)

            return get_lower_binary_impl(key, ty, impl)

    raise RuntimeError(f"Implementation of {fn} for {ty} not found")


__all__ = [
    "typing_registry",
    "impl_registry",
    "get_unary_impl_for_fn_and_ty",
    "get_binary_impl_for_fn_and_ty",
]
