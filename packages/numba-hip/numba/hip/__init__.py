from _ast import ImportFrom
from typing import Any
from numba import runtests
from numba.core import config

#: if config.ENABLE_CUDASIM:
#:     from .simulator_init import *
#: else:
#:     from .device_init import *
#:     from .device_init import _auto_device

#: from numba.cuda.compiler import compile_ptx, compile_ptx_for_current_device

#: def test(*args, **kwargs):
#:     if not is_available():
#:         raise cuda_error()

#:     return runtests.main("numba.cuda.tests", *args, **kwargs)

# ^ based on original code

from . import _modulerepl  # ! must come before subpackages that may use it

# -----------------------------------------------
# Derived modules, make local packages submodules
# -----------------------------------------------

import sys
import os
import re

from . import rocmpaths
from . import util

_mr = _modulerepl.ModuleReplicator(
    "numba.hip",
    os.path.join(os.path.dirname(__file__), "..", "cuda"),
    base_context=globals(),
    preprocess_all=lambda content: re.sub(
        r"\bnumba.cuda\b", "numba.hip", content
    ).replace("cudadrv", "hipdrv"),
)

api_util = _mr.create_and_register_derived_module(
    "api_util"
)  # make this a submodule of the package

from . import hipdrv

cudadrv = hipdrv
sys.modules["numba.hip.cudadrv"] = hipdrv
sys.modules["numba.hip.hipdrv"] = hipdrv

errors = _mr.create_and_register_derived_module(
    "errors",
    preprocess=lambda content: content.replace("CudaLoweringError", "HipLoweringError"),
)  # make this a submodule of the package

api = _mr.create_and_register_derived_module(
    "api"
)  # make this a submodule of the package

args = _mr.create_and_register_derived_module(
    "args"
)  # make this a submodule of the package

# Other
from .device_init import *
from .device_init import _auto_device

from . import codegen
from . import compiler

from .compiler import compile_llvm_ir, compile_llvm_ir_for_current_device

from . import decorators
from . import descriptor
from . import dispatcher
from . import target
from . import testing
from . import tests

# clean up
# del _preprocess
del sys
del os
del re
