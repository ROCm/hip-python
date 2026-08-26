# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

import clang.cindex


def _configure_libclang():
    """Set the libclang shared library before any tree.* import touches it.

    Default behavior: do **nothing** — let `clang.cindex` auto-load the
    `libclang.so` bundled inside the pip-installed `libclang` package
    (under `clang/native/libclang.so`). This pins the codegen to a
    well-tested libclang version that ships with the Python bindings.

    The previous implementation probed `/opt/rocm/lib/llvm/lib/libclang.so`
    and other system locations, which led to AST-shape mismatches when
    the system libclang was much newer than the pip wheel (the
    well-known anonymous-typedef enum spelling shift in libclang 17+
    showed up here as the `cyhipblas.pxd` `cdef enum hipblasStatus_t:`
    regression).

    Honors `INTERFACEGEN_LIBCLANG` if set, for advanced use only.
    """
    if clang.cindex.Config.loaded:
        return
    env = os.environ.get("INTERFACEGEN_LIBCLANG")
    if env and os.path.exists(env):
        clang.cindex.Config.set_library_file(env)
        return
    # Fall through: clang.cindex.Config auto-detects the libclang.so
    # bundled in the pip `libclang` package via Config.library_path.


_configure_libclang()
