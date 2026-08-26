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

"""Regression test: with ``python_interface_always_return_tuple`` enabled, a
single-return function's .pyi docstring must wrap the lone return value as a
size-1 result tuple, matching the .pyx (not the bare ``status: (undocumented)``
form).
"""

from interfacegen.test._codegen_helpers import find_function, make_generator

HEADER = r"""
typedef enum myStatus_t { MY_OK = 0 } myStatus_t;

/** \brief Does a thing. */
myStatus_t myDoThing(int x);
"""


def _stub_text(module_opts):
    root = make_generator(HEADER).backend.root
    fn = find_function(root, "myDoThing")
    return "\n".join(fn.render_pyi_stub("cy.", module_opts=module_opts))


def test_pyi_wraps_single_return_when_always_tuple():
    text = _stub_text({"python_interface_always_return_tuple": True})
    assert "Returns:" in text
    assert "tuple" in text
    assert "of size 1 that contains" in text


def test_pyi_unwrapped_single_return_by_default():
    text = _stub_text(None)
    assert "Returns:" in text
    assert "of size 1 that contains" not in text
