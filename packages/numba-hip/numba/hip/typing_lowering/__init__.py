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

import os
import re

import numba.hip._modulerepl as _modulerepl

_mr = _modulerepl.ModuleReplicator(
    "numba.hip.typing_lowering",
    os.path.join(os.path.dirname(__file__), "..", "..", "cuda"),
    base_context=globals(),
    preprocess_all=lambda content: re.sub(
        r"\bnumba.cuda\b", "numba.hip", content
    ).replace("cudadrv", "hipdrv"),
)

from . import stubs
from . import hipdevicelib

# Gives us types
#   Dim3(types.Type),
#   GridGroup(types.Type),
#   CUDADispatcher(types.Dispatcher)->HIPDispatcher(types.Dispatcher)
# Gives us global vars:
#   dim3 = Dim3(),
#   grid_group = GridGroup()
types = _mr.create_and_register_derived_module(
    "types", preprocess=lambda content: content.replace("CUDA", "HIP")
)  # make this a submodule of the package
delattr(types, "GridGroup")  # TODO cooperative groups
delattr(types, "grid_group")  # TODO cooperative groups

from . import models
from . import hip
from . import math
from . import numpy

ufuncs = _mr.create_and_register_derived_module(
    "ufuncs",
    preprocess=lambda content: content.replace(
        "numba.hip.mathimpl", "numba.hip.typing_lowering.math"
    ),  # NOTE the preprocess_all has converted numba.cuda.mathimpl -> numba.hip.mathimpl
)  # make this a submodule of the package

from . import vector_types

from .registries import (
    typing_registry,
    impl_registry,
)
