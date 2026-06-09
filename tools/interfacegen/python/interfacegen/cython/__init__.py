"""Cython backend for interfacegen.

Public API is the union of names re-exported here. Submodules are
implementation detail (prefixed `_`); use `from interfacegen.cython
import X`, not `from interfacegen.cython._mixins import X`.
"""
# Re-export parent-package modules that the historical monolithic
# cython.py made available as `cython.tree`, `cython.cparser`, etc.
# Treefactory and a few tests reach for these through the cython
# namespace, so preserving them here keeps every existing import path
# working without touching call sites.
from .. import cparser, cythontemplates, doxyparser, tree  # noqa: F401
from ..support import cython as support  # noqa: F401
from ..support.recipes import control  # noqa: F401

from ._defaults import *  # noqa: F401,F403
from ._doxygen import *   # noqa: F401,F403
from ._mixins import *    # noqa: F401,F403
from ._entities import *  # noqa: F401,F403
from ._function import *  # noqa: F401,F403
from ._backend import *   # noqa: F401,F403
