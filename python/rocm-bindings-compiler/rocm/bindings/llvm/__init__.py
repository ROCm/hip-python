# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

"""LLVM C API bindings for ROCm.

This package provides Python bindings to the LLVM C API using ROCm's LLVM installation.

Library Discovery:
    The LLVM library (libLLVM.so) is discovered lazily when first accessed via
    rocm.bindings.util.paths.get_library_path(). The library is either:
    - Bundled in the wheel at rocm/bindings/llvm/libLLVM.so (when system version unavailable)
    - Loaded from system ROCm LLVM installation (/opt/rocm/llvm/lib)
    - Loaded via rocm_sdk (TheRock) if installed
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# Import subpackages
# Each subpackage's Cython modules will lazy-load libLLVM.so on first function call
from rocm.bindings.llvm import c, config

__all__ = ['c', 'config']
