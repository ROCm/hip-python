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

__author__ = "Advanced Micro Devices, Inc."

import enum
import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import tree


def fallback(*fallbacks):
    """Chain rule callables: the wrapped function runs first; if it returns
    None, each callable in ``fallbacks`` is tried in order. First non-None
    wins; if every callable returns None, the chain returns None.

    Intended for the per-library ``ptr_parm_intent`` / ``ptr_rank`` rules in
    ``support.recipes.rocm`` so library-specific overrides delegate to
    ``support.recipes.generic`` (modifier-based deductions) and finally to
    ``DEFAULT_PTR_PARM_INTENT`` / ``DEFAULT_PTR_RANK``.

    Decorator order: ``@staticmethod`` outside, ``@fallback(...)`` inside, so
    ``fallback`` wraps the plain function before ``staticmethod`` turns it
    into a descriptor::

        class hip:
            @staticmethod
            @fallback(generic.conservative.ptr_parm_intent,
                      DEFAULT_PTR_PARM_INTENT)
            def ptr_parm_intent(parm):
                if (parm.parent.name, parm.parm_index) == ("hipDeviceGetName", 0):
                    return ParmIntent.OUT
                return None  # defer
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(node):
            r = fn(node)
            if r is not None:
                return r
            for fb in fallbacks:
                r = fb(node)
                if r is not None:
                    return r
            return None

        return wrapper

    return deco


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
    from interfacegen import tree

    assert isinstance(node, tree.Typed)
    # Every pointer defaults to rank 1. ``char *`` is intentionally NOT
    # special-cased to rank 0 here: a NUL-terminated string is rank-1
    # data (see ``generic.string_z``), and the single-char-by-reference
    # case is rare enough to require an explicit per-parameter rank-0
    # override. The scalar-vs-string / IN-vs-OUT distinction for char
    # pointers is made downstream in the complicated-type handler.
    return 1
