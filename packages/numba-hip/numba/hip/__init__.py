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
from .typing_lowering import stubs
from .typing_lowering import hipdevicelib
from .typing_lowering import math
from .typing_lowering import numpy

mr = _modulerepl.ModuleReplicator(
    "numba.hip",
    os.path.join(os.path.dirname(__file__), "..", "cuda"),
    base_context=globals(),
    preprocess_all=lambda content: re.sub(
        r"\bnumba.cuda\b", "numba.hip", content
    ).replace("cudadrv", "hipdrv"),
)

api_util = mr.create_and_register_derived_module(
    "api_util"
)  # make this a submodule of the package

from . import hipdrv

cudadrv = hipdrv
sys.modules["numba.hip.cudadrv"] = hipdrv
sys.modules["numba.hip.hipdrv"] = hipdrv

errors = mr.create_and_register_derived_module(
    "errors",
    preprocess=lambda content: content.replace("CudaLoweringError", "HipLoweringError"),
)  # make this a submodule of the package

api = mr.create_and_register_derived_module(
    "api"
)  # make this a submodule of the package

args = mr.create_and_register_derived_module(
    "args"
)  # make this a submodule of the package

# Gives us types
#   Dim3(types.Type),
#   GridGroup(types.Type),
#   CUDADispatcher(types.Dispatcher)->HIPDispatcher(types.Dispatcher)
# Gives us global vars:
#   dim3 = Dim3(),
#   grid_group = GridGroup()
types = mr.create_and_register_derived_module(
    "types", preprocess=lambda content: content.replace("CUDA", "HIP")
)  # make this a submodule of the package

# TODO reenable models
# models = mr.create_and_register_derived_module(
#     "models",
#     preprocess=_preprocess
# )  # make this a submodule of the package

# TODO: reuse mathimpl as it delegates only to libdevice, hipdevicelib is our superset
# replace: from numba.cuda import libdevice -> numba.hip import hipdeviceimpl
#          from numba import cuda -> from numba import hip as cuda # required for cuda.fp16... expressions

# Other
from .device_init import *
from .device_init import _auto_device

# clean up
# del _preprocess
del mr
del _modulerepl
del sys
del os
del re
