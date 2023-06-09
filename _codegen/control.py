# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

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