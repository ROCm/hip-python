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
# Modifications Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# isort: skip_file
# fmt: off

import numbers
from numba.core.errors import LoweringError


class KernelRuntimeError(RuntimeError):
    def __init__(self, msg, tid=None, ctaid=None):
        self.tid = tid
        self.ctaid = ctaid
        self.msg = msg
        t = ("An exception was raised in thread=%s block=%s\n"
             "\t%s")
        msg = t % (self.tid, self.ctaid, self.msg)
        super(KernelRuntimeError, self).__init__(msg)


class HipLoweringError(LoweringError):
    pass


_launch_help_url = ("https://numba.readthedocs.io/en/stable/cuda/"
                    "kernels.html#kernel-invocation")
missing_launch_config_msg = """
Kernel launch configuration was not specified. Use the syntax:

kernel_function[blockspergrid, threadsperblock](arg0, arg1, ..., argn)

See {} for help.

""".format(_launch_help_url)


def normalize_kernel_dimensions(griddim, blockdim):
    """
    Normalize and validate the user-supplied kernel dimensions.
    """

    def check_dim(dim, name):
        if not isinstance(dim, (tuple, list)):
            dim = [dim]
        else:
            dim = list(dim)
        if len(dim) > 3:
            raise ValueError('%s must be a sequence of 1, 2 or 3 integers, '
                             'got %r' % (name, dim))
        for v in dim:
            if not isinstance(v, numbers.Integral):
                raise TypeError('%s must be a sequence of integers, got %r'
                                % (name, dim))
        while len(dim) < 3:
            dim.append(1)
        return tuple(dim)

    if None in (griddim, blockdim):
        raise ValueError(missing_launch_config_msg)

    griddim = check_dim(griddim, 'griddim')
    blockdim = check_dim(blockdim, 'blockdim')

    return griddim, blockdim
