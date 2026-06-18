# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
