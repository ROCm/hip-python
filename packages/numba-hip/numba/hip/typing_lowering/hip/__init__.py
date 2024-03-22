# MIT License
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

"""Typing declarations and lowering impls for HIP types

Typing declarations and lowering impls for HIP types such as 
dim3 and arrays in GPU address spaces.

Attributes:
    typing_registry (`numba.core.typing.templates.Registry`):
        A registry of typing declarations. The registry stores such declarations
        for functions, attributes and globals.
    impl_registry (`numba.core.imputils.Registry`):
        A registry of function and attribute implementations.
"""

from numba.hip.typing_lowering.stubs import Stub

from . import hipstubs

# Expose vector type constructors and aliases as module level attributes.
thestubs = {}
for vector_type_stub in hipstubs._vector_type_stubs:
    thestubs[vector_type_stub.__name__] = vector_type_stub
    for alias in vector_type_stub.aliases:
        thestubs[alias] = vector_type_stub
del vector_type_stub
# print(vars(hipstubs))
for k, v in vars(hipstubs).items():
    try:
        if issubclass(v, Stub):
            thestubs[k] = v
    except TypeError:
        # 'v' must be a class to use 'issubclass'
        pass
globals().update(thestubs)

from . import typing
from . import lowering
