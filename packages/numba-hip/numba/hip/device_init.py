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

# Re export
import sys
#: from .stubs import (threadIdx, blockIdx, blockDim, gridDim, laneid, warpsize,
#:                     syncwarp, shared, local, const, atomic,
#:                     shfl_sync_intrinsic, vote_sync_intrinsic, match_any_sync, #: TODO: HIP/AMD: provide the correct intrinsics, support trivial _sync intrinsics
#:                     match_all_sync, threadfence_block, threadfence_system,
#:                     threadfence, selp, popc, brev, clz, ffs, fma, cbrt, cg,
#:                     activemask, lanemask_lt, nanosleep, fp16,
#:                     _vector_type_stubs)
from .typing_lowering.stubs import (
    _vector_type_stubs,
    shared,
    local,
    const,
)
from .typing_lowering import hipdevicelib
globals().update(hipdevicelib.stubs)
del globals()["gridsize"] # will be imported as intrinsic below
del globals()["warpsize"] # hipdevicelib implements it as function, intrinsics adds it as attribute
del hipdevicelib

#: from .intrinsics import (grid, gridsize, syncthreads, syncthreads_and,
#:                          syncthreads_count, syncthreads_or)
from .intrinsics import (grid, gridsize)
from .hipdrv.error import HipSupportError
from .hipdrv.error import HipSupportError as CudaSupportError
from .hipdrv.driver import (BaseHIPMemoryManager,
                            HostOnlyHIPMemoryManager,
                            GetIpcHandleMixin, MemoryPointer,
                            MappedMemory, PinnedMemory, MemoryInfo,
                            IpcHandle, set_memory_manager)
from numba.cuda.cudadrv.runtime import runtime
#: from .cudadrv import nvvm #: FIXME: HIP/AMD: not supported
from numba.hip import initialize
from .errors import KernelRuntimeError

from .decorators import jit, declare_device
from .api import *
from .api import _auto_device
from .args import In, Out, InOut

#: from .kernels import reduction #: FIXME: HIP/AMD: not supported yet

#: reduce = Reduce = reduction.Reduce #: FIXME: HIP/AMD: not supported yet

# Expose vector type constructors and aliases as module level attributes.
for vector_type_stub in _vector_type_stubs:
    setattr(sys.modules[__name__], vector_type_stub.__name__, vector_type_stub)
    for alias in vector_type_stub.aliases:
        setattr(sys.modules[__name__], alias, vector_type_stub)
del vector_type_stub, _vector_type_stubs


def is_available():
    """Returns a boolean to indicate the availability of a CUDA GPU.

    This will initialize the driver if it hasn't been initialized.
    """
    # whilst `driver.is_available` will init the driver itself,
    # the driver initialization may raise and as a result break
    # test discovery/orchestration as `cuda.is_available` is often
    # used as a guard for whether to run a CUDA test, the try/except
    # below is to handle this case.
    driver_is_available = False
    try:
        driver_is_available = driver.driver.is_available
    except HipSupportError:
        pass

    return driver_is_available #:  and nvvm.is_available() #: TODO: HIP/AMD: not supported yet


def is_supported_version():
    """Returns True if the CUDA Runtime is a supported version.

    Unsupported versions (e.g. newer versions than those known to Numba)
    may still work; this function provides a facility to check whether the
    current Numba version is tested and known to work with the current
    runtime version. If the current version is unsupported, the caller can
    decide how to act. Options include:

    - Continuing silently,
    - Emitting a warning,
    - Generating an error or otherwise preventing the use of CUDA.
    """

    return runtime.is_supported_version()


def cuda_error():
    """Returns None if there was no error initializing the CUDA driver.
    If there was an error initializing the driver, a string describing the
    error is returned.
    """
    return driver.driver.initialization_error # driver avail via 'from api import *'


initialize.initialize_all()
