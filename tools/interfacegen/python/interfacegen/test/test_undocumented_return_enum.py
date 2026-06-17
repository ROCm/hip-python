# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
    returns_block = docstring[docstring.index("Returns:"):]
    # the undocumented status enum must appear, and before the out argument
    assert "myStatus_t" in returns_block
    assert "tuple" in returns_block
    status_pos = returns_block.index("myStatus_t")
    out_pos = returns_block.index("where the handle is stored")
    assert status_pos < out_pos, f"status enum not first:\n{returns_block}"
