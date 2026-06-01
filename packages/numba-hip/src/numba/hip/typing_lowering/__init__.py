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

# flake8: noqa
# isort: skip_file
# fmt: off

# NOTE: order is important
from . import (
    stubs,
    hipdevicelib,
    types,
    hip,
    math,
    models,
    numpy,
    ufuncs,
    vector_types,
)
from .registries import (
    impl_registry,
    typing_registry,
)

# `types` gives us types
#   Dim3(types.Type),
#   GridGroup(types.Type),
#   CUDADispatcher(types.Dispatcher)->HIPDispatcher(types.Dispatcher)
# `types` gives us global vars:
#   dim3 = Dim3(),
#   grid_group = GridGroup()
delattr(types, "GridGroup")  # TODO cooperative groups
delattr(types, "grid_group")  # TODO cooperative groups
