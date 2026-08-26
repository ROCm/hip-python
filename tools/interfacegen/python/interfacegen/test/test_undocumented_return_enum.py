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

"""Regression test: a non-void function whose return value (typically a status
enum) is not documented must still list that return as the first entry of the
documented return tuple, matching the runtime ``(retval, *out_args)`` order.
"""

from interfacegen.test._codegen_helpers import find_function, make_generator

HEADER = r"""
typedef enum myStatus_t { MY_OK = 0 } myStatus_t;

/** \brief Allocates a thing.
 *
 * \param[out] out where the handle is stored
 */
myStatus_t myAlloc(void** out);
"""


def test_undocumented_return_enum_is_first_in_tuple():
    # make_generator-backed tree so the Function node has raw_comment_cleaner
    # set (build_root alone skips that initialization).
    root = make_generator(HEADER).backend.root
    fn = find_function(root, "myAlloc")

    # 'out' is a returned (callee-written) argument; the return value itself
    # has no @return doxygen section.
    docstring = fn._render_python_docstring(
        out_arg_names=["out"],
        parm_python_types={"out": "object"},
    )

    assert "Returns:" in docstring
    returns_block = docstring[docstring.index("Returns:") :]
    # the undocumented status enum must appear, and before the out argument
    assert "myStatus_t" in returns_block
    assert "tuple" in returns_block
    status_pos = returns_block.index("myStatus_t")
    out_pos = returns_block.index("where the handle is stored")
    assert status_pos < out_pos, f"status enum not first:\n{returns_block}"
