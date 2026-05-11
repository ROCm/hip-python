# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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


"""Cython backend — doxygen submodule.

Carved out of the historical interfacegen.cython monolith.
Public callers should import from `interfacegen.cython`,
not from this submodule directly.
"""

__author__ = "Advanced Micro Devices, Inc."

import ctypes
import keyword
import logging
import os
import re
import sys
import textwrap
import typing

import clang.cindex
import Cython.Tempita

from .. import cparser, cythontemplates, doxyparser, tree
from ..support import cython as support
from ..support.recipes import control

_log = logging.getLogger("interfacegen")
from . import _defaults
from ._defaults import *  # noqa: F401,F403

__all__ = [
    'DOXYGEN_CONV',
]

# doxygen parser
DOXYGEN_CONV = doxyparser.DoxygenGrammar()
DOXYGEN_CONV.escaped.set_parse_action(doxyparser.format.PythonDocstrings.escaped)
DOXYGEN_CONV.with_word.set_parse_action(
    doxyparser.format.PythonDocstrings.with_word
)
DOXYGEN_CONV.fdollar.set_parse_action(doxyparser.format.PythonDocstrings.fdollar)
DOXYGEN_CONV.frnd.set_parse_action(doxyparser.format.PythonDocstrings.frnd)


def reference_(tokens):
    global python_interface_pyobj_role_template
    reference: str = tokens[0].replace("#", ".")
    reference = reference.replace("::", ".")
    return python_interface_pyobj_role_template.format(
        name=reference.lstrip(".")
    )


DOXYGEN_CONV.see_reference.set_parse_action(reference_)
DOXYGEN_CONV.in_text_reference.set_parse_action(reference_)


def other_parse_action(tokens):
    cmd = tokens[0][1:]
    if cmd == "ref":
        return f"``{tokens[1]}`` "
    return []  # suppress all others


DOXYGEN_CONV.other.set_parse_action(other_parse_action)

# Mixins

