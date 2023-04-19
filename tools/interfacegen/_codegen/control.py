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


class Intent(enum.IntEnum):
    IN = 0
    OUT = 1
    INOUT = 2
    CREATE = 3  # OUT result that is also created


class Rank(enum.IntEnum):
    SCALAR = 0
    ARRAY = 1
    ANY = 2

    @property
    def might_be_array(self):
        """Returns if this ``Rank`` would match an array.
        """
        return self.value in (Rank.ARRAY.value,Rank.ANY.value)


def DEFAULT_PTR_PARM_INTENT(node: "tree.Node"):
    return Intent.INOUT


def DEFAULT_PTR_RANK(node: "tree.Node"):
    return Rank.ANY
