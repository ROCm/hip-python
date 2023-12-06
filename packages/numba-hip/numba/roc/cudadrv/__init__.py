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

import numba.roc._modulerepl as _modulerepl
import os
import re

mr = _modulerepl.ModuleReplicator(
    "numba.roc.cudadrv", os.path.join(os.path.dirname(__file__), "..", "..", "cuda", "cudadrv"),
    base_context=globals(),
    preprocess_all=lambda content: re.sub(r"\bnumba.cuda\b","numba.roc",content),
)
 
# order is important here!

_extras = mr.create_and_register_derived_module(
    "_extras",
    from_file=False,
    module_content="CUDA_IPC_HANDLE_SIZE=HIP_IPC_HANDLE_SIZE=64\n"
)  # make this a submodule of the package

enums = mr.create_and_register_derived_module(
    "enums"
)  # make this a submodule of the package

error = mr.create_and_register_derived_module(
    "error"
)  # make this a submodule of the package

drvapi = mr.create_and_register_derived_module(
    "drvapi",
)  # make this a submodule of the package
for k,v in list(drvapi.__dict__.items()):
    if k.startswith("cu_"):
        drvapi.__dict__["hip_"+k[3:]] = v
    elif k.startswith("CU_"):
        drvapi.__dict__["HIP_"+k[3:]] = v
    elif k == "API_PROTOTYPES":
        for api,args in list(v.items()):
            # FIXME some of the API are not supported by HIP
            assert api.startswith("cu")
            v["hip"+api[2:]] = args

rtapi = mr.create_and_register_derived_module(
    "rtapi",
)  # make this a submodule of the package
for api,args in list(rtapi.__dict__["API_PROTOTYPES"].items()):
    # FIXME some of the API are not supported by HIP
    assert api.startswith("cuda")
    v["hip"+api[2:]] = args

nvrtc = mr.create_and_register_derived_module(
    "nvrtc",
)  # make this a submodule of the package

from . import driver

devices = mr.create_and_register_derived_module(
    "devices"
)  # make this a submodule of the package

devicearray = mr.create_and_register_derived_module(
    "devicearray"
)  # make this a submodule of the package

ndarray = mr.create_and_register_derived_module(
    "ndarray"
)  # make this a submodule of the package

from . import nvvm

# clean up
del mr
del _modulerepl
del os
del re
