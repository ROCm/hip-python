# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import enum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import tree


class Warnings(enum.IntEnum):
    IGNORE = 0
    WARN = 1
    ERROR = 2

def DEFAULT_RENAMER(name: str):
    return name

def DEFAULT_NODE_FILTER(node: "tree.Node"):
    return True

class ParmIntent(enum.IntEnum):
    NONE = -1
    IN = 0
    INOUT = 1
    OUT = 2

def DEFAULT_PTR_PARM_INTENT(node: "tree.Parm"):
    if node.is_double_pointer_to_non_const_type:
        return ParmIntent.INOUT

RANK_ANY = -1

def DEFAULT_PTR_RANK(node: "tree.Node"):
    from . import tree
    
    assert isinstance(node,tree.Typed)
    if node.is_pointer_to_char():
        return 0
    return 1