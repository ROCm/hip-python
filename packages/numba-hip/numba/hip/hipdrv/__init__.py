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

"""HIP-based ROC Driver

- Driver API binding
- Device array implementation

"""

from numba.core import config
assert not config.ENABLE_CUDASIM, "Cannot use real driver API with simulator"

# ^ based on original code

#-----------------------
# Now follow the modules
#-----------------------

import numba.hip.util.modulerepl as _modulerepl
import os
import re

mr = _modulerepl.ModuleReplicator(
    "numba.hip.hipdrv", os.path.join(os.path.dirname(__file__), "..", "..", "cuda", "cudadrv"),
    base_context=globals(),
    preprocess_all=lambda content: re.sub(r"\bnumba.cuda\b","numba.hip",content).replace("cudadrv","hipdrv"),
)
 
# order is important here!

from . import _extras

from . import hiprtc
nvrtc = hiprtc

from . import driver

# DOCS 'devices':
# Expose each GPU devices directly.
# 
# This module implements a API that is like the "HIP runtime" context manager
# for managing HIP context stack and clean up.  It relies on thread-local globals
# to separate the context stack management of each thread. Contexts are also
# shareable among threads.  Only the main thread can destroy Contexts.
# 
# Note:
# - This module must be imported by the main-thread.

devices = mr.create_and_register_derived_module(
    "devices",
)  # make this a submodule of the package

devicearray = mr.create_and_register_derived_module(
    "devicearray",
    preprocess=lambda content: content.replace("from numba import cuda","from numba import hip as cuda"),
    # preprocess=lambda content: content.replace("CUDA","HIP")
    # Reuse CUDA config values as they are for now: config.CUDA_WARN_ON_IMPLICIT_COPY
)  # make this a submodule of the package

ndarray = mr.create_and_register_derived_module(
    "ndarray",
)  # make this a submodule of the package

# clean up
del mr
del _modulerepl
del os
del re
