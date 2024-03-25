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

from numba import hip
hip.pose_as_cuda()

# unchanged original unit Numba CUDA test code below:

from numba import hip
hip.pose_as_cuda()

from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase

import numpy as np
from numba import config, cuda, njit, types


class Interval:
    """
    A half-open interval on the real number line.
    """
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return 'Interval(%f, %f)' % (self.lo, self.hi)

    @property
    def width(self):
        return self.hi - self.lo


@njit
def interval_width(interval):
    return interval.width


@njit
def sum_intervals(i, j):
    return Interval(i.lo + j.lo, i.hi + j.hi)


if not config.ENABLE_CUDASIM:
    from numba.core import cgutils
    from numba.core.extending import (lower_builtin, make_attribute_wrapper,
                                      models, register_model, type_callable,
                                      typeof_impl)
    from numba.core.typing.templates import AttributeTemplate
    from numba.cuda.cudadecl import registry as cuda_registry
    from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr

    class IntervalType(types.Type):
        def __init__(self):
            super().__init__(name='Interval')

    interval_type = IntervalType()

    @typeof_impl.register(Interval)
    def typeof_interval(val, c):
        return interval_type

    @type_callable(Interval)
    def type_interval(context):
        def typer(lo, hi):
            if isinstance(lo, types.Float) and isinstance(hi, types.Float):
                return interval_type
        return typer

    @register_model(IntervalType)
    class IntervalModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [
                ('lo', types.float64),
                ('hi', types.float64),
            ]
            models.StructModel.__init__(self, dmm, fe_type, members)

    make_attribute_wrapper(IntervalType, 'lo', 'lo')
    make_attribute_wrapper(IntervalType, 'hi', 'hi')

    @lower_builtin(Interval, types.Float, types.Float)
    def impl_interval(context, builder, sig, args):
        typ = sig.return_type
        lo, hi = args
        interval = cgutils.create_struct_proxy(typ)(context, builder)
        interval.lo = lo
        interval.hi = hi
        return interval._getvalue()

    @cuda_registry.register_attr
    class Interval_attrs(AttributeTemplate):
        key = IntervalType

        def resolve_width(self, mod):
            return types.float64

    @cuda_lower_attr(IntervalType, 'width')
    def cuda_Interval_width(context, builder, sig, arg):
        lo = builder.extract_value(arg, 0)
        hi = builder.extract_value(arg, 1)
        return builder.fsub(hi, lo)


@skip_on_cudasim('Extensions not supported in the simulator')
class TestExtending(CUDATestCase):
    def test_attributes(self):
        @cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = iv.lo
            r[1] = iv.hi

        x = np.asarray((1.5, 2.5))
        r = np.zeros_like(x)

        f[1, 1](r, x)

        np.testing.assert_equal(r, x)

    def test_property(self):
        @cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = iv.width

        x = np.asarray((1.5, 2.5))
        r = np.zeros(1)

        f[1, 1](r, x)

        np.testing.assert_allclose(r[0], x[1] - x[0])

    def test_extension_type_as_arg(self):
        @cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = interval_width(iv)

        x = np.asarray((1.5, 2.5))
        r = np.zeros(1)

        f[1, 1](r, x)

        np.testing.assert_allclose(r[0], x[1] - x[0])

    def test_extension_type_as_retvalue(self):
        @cuda.jit
        def f(r, x):
            iv1 = Interval(x[0], x[1])
            iv2 = Interval(x[2], x[3])
            iv_sum = sum_intervals(iv1, iv2)
            r[0] = iv_sum.lo
            r[1] = iv_sum.hi

        x = np.asarray((1.5, 2.5, 3.0, 4.0))
        r = np.zeros(2)

        f[1, 1](r, x)

        expected = np.asarray((x[0] + x[2], x[1] + x[3]))
        np.testing.assert_allclose(r, expected)


if __name__ == '__main__':
    unittest.main()
