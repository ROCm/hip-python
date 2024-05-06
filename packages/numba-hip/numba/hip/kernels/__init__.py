# MIT License
#
# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

import numba.hip.util.modulerepl as _modulerepl

_mr = _modulerepl.ModuleReplicator(
    "numba.hip.kernels",
    os.path.join(os.path.dirname(__file__), "..", "..", "cuda", "kernels"),
    base_context=globals(),
    preprocess_all=lambda content: re.sub(
        r"from\s+numba\s+import\s+cuda", "from numba import hip as cuda", content
    ).replace("numba.cuda","numba.hip"),
)

reduction = _mr.create_and_register_derived_module(
    "reduction",
    preprocess=lambda content: re.sub(r"(_WARPSIZE\s*=\s*)[0-9]+", r"\g<1>64", content),
)  # make this a submodule of the package

transpose = _mr.create_and_register_derived_module(
    "transpose"
)  # make this a submodule of the package

# clean up
del re
del os
del _mr
