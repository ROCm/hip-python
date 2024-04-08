"""HIP-based ROC Driver

- Driver API binding
- Device array implementation

"""

from numba.core import config
assert not config.ENABLE_CUDASIM, "Cannot use real driver API with simulator"

# ^ based on original code

#-----------------------
# Now follow the modules
#-----------------------

import numba.hip.util.modulerepl as _modulerepl
import os
import re

mr = _modulerepl.ModuleReplicator(
    "numba.hip.hipdrv", os.path.join(os.path.dirname(__file__), "..", "..", "cuda", "cudadrv"),
    base_context=globals(),
    preprocess_all=lambda content: re.sub(r"\bnumba.cuda\b","numba.hip",content).replace("cudadrv","hipdrv"),
)
 
# order is important here!

from . import _extras

from . import hiprtc
nvrtc = hiprtc

from . import driver

# DOCS 'devices':
# Expose each GPU devices directly.
# 
# This module implements a API that is like the "HIP runtime" context manager
# for managing HIP context stack and clean up.  It relies on thread-local globals
# to separate the context stack management of each thread. Contexts are also
# shareable among threads.  Only the main thread can destroy Contexts.
# 
# Note:
# - This module must be imported by the main-thread.

devices = mr.create_and_register_derived_module(
    "devices",
)  # make this a submodule of the package

devicearray = mr.create_and_register_derived_module(
    "devicearray",
    preprocess=lambda content: content.replace("from numba import cuda","from numba import hip as cuda"),
    # preprocess=lambda content: content.replace("CUDA","HIP")
    # Reuse CUDA config values as they are for now: config.CUDA_WARN_ON_IMPLICIT_COPY
)  # make this a submodule of the package

ndarray = mr.create_and_register_derived_module(
    "ndarray",
)  # make this a submodule of the package

# clean up
del mr
del _modulerepl
del os
del re
