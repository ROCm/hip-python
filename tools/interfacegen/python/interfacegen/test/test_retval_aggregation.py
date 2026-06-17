# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Regression test: ``\\retval`` / ``@retval`` entries describe the C return
value (the first tuple entry). They must be aggregated into that return value's
description, preserving each value name, instead of being rendered as separate
"Retval:" sections (which dropped the value name).
"""

from interfacegen.test._codegen_helpers import find_function, make_generator

HEADER = r"""
typedef enum myStatus_t { MY_OK = 0 } myStatus_t;

/** \brief Does a thing.
 *
 * \param[in] x an input value
 * \retval MY_STATUS_OK When everything is fine.
 * \retval `MY_STATUS_BAD` When something is wrong.
 */
myStatus_t myDo(int x);
"""


def test_retval_entries_aggregate_into_return_value():
    root = make_generator(HEADER).backend.root
    fn = find_function(root, "myDo")

    docstring = fn._render_python_docstring(
        out_arg_names=[],
        parm_python_types={"x": "int"},
    )

    assert "Returns:" in docstring
    returns_block = docstring[docstring.index("Returns:"):]

    # The return typename heads the aggregated description.
    assert "myStatus_t" in returns_block
    # Both retval names survive (bare and backtick-quoted forms alike), and
    # are rendered as cross-reference roles, not dropped.
    assert "MY_STATUS_OK" in returns_block
    assert "MY_STATUS_BAD" in returns_block
    assert ":py:obj:`.MY_STATUS_OK`" in returns_block
    assert ":py:obj:`.MY_STATUS_BAD`" in returns_block
    # Their descriptions are aggregated alongside the names.
    assert "When everything is fine." in returns_block
    assert "When something is wrong." in returns_block
    # Multiple entries render as a "One of:" list under the return value.
    assert "One of:" in returns_block

    # No standalone "Retval:" section leaks anywhere in the docstring, and no
    # raw @retval tag survives.
    assert "Retval:" not in docstring
    assert "@retval" not in docstring
