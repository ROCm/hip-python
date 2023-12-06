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

mr = _modulerepl.ModuleReplicator(
    "numba.roc",
    os.path.join(os.path.dirname(__file__), "..", "cuda"),
    base_context=globals(),
    preprocess_all=lambda content: re.sub(r"\bnumba.cuda\b", "numba.roc", content),
)

api_util = mr.create_and_register_derived_module(
    "api_util"
)  # make this a submodule of the package

from . import cudadrv
hipdrv = cudadrv  # some might prefer that name
sys.modules["numba.roc.hipdrv"] = hipdrv

errors = mr.create_and_register_derived_module(
    "errors"
)  # make this a submodule of the package

api = mr.create_and_register_derived_module(
    "api"
)  # make this a submodule of the package

args = mr.create_and_register_derived_module(
    "args"
)  # make this a submodule of the package

types = mr.create_and_register_derived_module(
    "types"
)  # make this a submodule of the package

# FIXME remove the _sync stubs?
# FIXME add non-_sync stubs?
stubs = mr.create_and_register_derived_module(
    "stubs"
)  # make this a submodule of the package

vector_types = mr.create_and_register_derived_module(
    "vector_types"
)  # make this a submodule of the package

def _preprocess(content: str):
    import ast
    key1  = mr.to_ast_node("from numba import cuda")

    class Transformer(ast.NodeTransformer):
        def visit_ImportFrom(self, node: ImportFrom) -> Any:
            nonlocal key1
            if mr.compare_ast_nodes(node,key1):
                return mr.to_ast_node("from numba import roc as cuda")
            return node
        
    result = Transformer().visit(
        compile(content, "<string>", "exec", ast.PyCF_ONLY_AST)
    )
    #print(ast.unparse(result))
    return result

# cudadecl = mr.create_and_register_derived_module(
#     "cudadecl",
#     preprocess=_preprocess
# )  # make this a submodule of the package
# FIXME 'from numba.cuda.compiler import declare_device_function_template'

cudamath = mr.create_and_register_derived_module(
    "cudamath"
)  # make this a submodule of the package

vectorizers = mr.create_and_register_derived_module(
    "vectorizers",
    preprocess=_preprocess
)  # make this a submodule of the package

libdevice = mr.create_and_register_derived_module(
    "libdevice"
)  # make this a submodule of the package

libdevicefuncs = mr.create_and_register_derived_module(
    "libdevicefuncs"
)  # make this a submodule of the package
# FIXME All routines have __nv_ prefix

# libdevicedecl = mr.create_and_register_derived_module(
#     "libdevicedecl"
# )  # make this a submodule of the package
# FIXME 'AttributeError: partially initialized module 'numba.roc' has no attribute 'abs' (most likely due to a circular import)'

# libdeviceimpl = mr.create_and_register_derived_module(
#     "libdeviceimpl"
# )  # make this a submodule of the package
# FIXME 'AttributeError: partially initialized module 'numba.roc' has no attribute 'abs' (most likely due to a circular import)'

mathimpl = mr.create_and_register_derived_module(
    "mathimpl",
    preprocess=_preprocess
)  # make this a submodule of the package

models = mr.create_and_register_derived_module(
    "models",
    preprocess=_preprocess
)  # make this a submodule of the package

# Other
from .device_init import *
from .device_init import _auto_device

from . import nvvmutils

# clean up
del _preprocess
del mr
del _modulerepl
del sys
del os
del re
