# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Shared setup for the tests of the generators themselves.

Tests that drive a real generator reuse the scaffolding of the interfacegen
suite (`_codegen_helpers.make_generator` and friends). That directory holds
no `__init__.py` and is not part of the installed `interfacegen` package, so
it is reached through the checkout rather than through an import.
"""

import os
import sys

import clang.cindex

INTERFACEGEN_TEST_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "interfacegen",
        "python",
        "interfacegen",
        "test",
    )
)

if INTERFACEGEN_TEST_DIR not in sys.path:
    sys.path.insert(0, INTERFACEGEN_TEST_DIR)


def _configure_libclang():
    """Pick the same libclang the interfacegen suite runs against.

    Doing nothing loads the `libclang.so` bundled in the pip-installed
    `libclang` package, which is the version the codegen is tested with;
    `INTERFACEGEN_LIBCLANG` overrides it. See the interfacegen conftest for
    why probing system locations instead led to AST-shape mismatches.
    """
    if clang.cindex.Config.loaded:
        return
    env = os.environ.get("INTERFACEGEN_LIBCLANG")
    if env and os.path.exists(env):
        clang.cindex.Config.set_library_file(env)


_configure_libclang()
