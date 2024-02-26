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

from . import hipconfig
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


def current_hip_extra_cflags():
    """Returns current HIP device library compiler flags."""
    from numba.hip.typing_lowering.hipdevicelib import hipdevicelib

    return list(hipdevicelib.USER_HIP_CFLAGS)  # copy


def current_hip_extensions():
    """Returns current HIP device library extensions.

    Returns the current user-specified HIP device library extensions
    as `str`.
    """
    from numba.hip.typing_lowering.hipdevicelib import hipdevicelib

    return str(hipdevicelib.USER_HIP_EXTENSIONS)  # copy


def set_hip_extensions(
    code=None,
    filepath=None,
    extra_cflags=None,
):
    """Extend Numba HIP device libray via HIP C++.

    (EXPERIMENTAL)

    Extend the Numba HIP device library with your own device functions
    expressed in C++. The functions can then used in kernels
    and device functions via `hip.<func_name>`.

    Limitation:
        Currently, only functions that take scalar attributes of primitive type can be added.
        Any pointer arguments are interpreted as scalar return values.
        (TODO HIP workaround this limitation by allowing to supply a pointer intent callback/map.
        Alternatively, user could cast pointers to 64-bit integers in Numba HIP Python code and manually cast
        back to pointer types in user function body.)
        (TODO HIP support arrays by providing intrinsic that gets pointer as 64-bit int from (array|struct)( element)?)
        (TODO HIP add some sort of filtering, likely not all functions in a translation unit should be wrapped.
        Alternatively, only accept the functions directly in the specified source and not
        those in any dependencies? File association can be checked via clang cursor
        location metadata.)

    Note:
        At least and only one of arguments 'code'
        and 'filepath' must not be ``None``.
    Note:
        The provided HIP C++ device functions can have inline hints or
        enforcing attributes. A wrapper function is generated to make
        them available in this case.
    Note:
        Your extensions are included into a compilation unit
        that includes the default HIP C++ library source
        first and then puts your extensions below it.
    Note:
        You can append include paths, definitions and other
        compiler flags via option `extra_cflags`.

    Warning:
        Every extension of the Numba HIP device library
        requires and triggers a recompilation, which may take
        a few seconds.

    Args:
        code (`str`, optional):
            Contents of an HIP C++ source file that will
            be set as extensions of
            Mutually exclusive with `filepath`.
            One must be specified.
            Defaults to ``None``.
        filepath (`str`, optional):
            Path to an HIP C++ source file.
            Mutually exclusive with `code`.
            One must be specified.
            Defaults to ``None``.
        extra_cflags (`list` or `None`):
            Compilation flags that are appended to
            the flags of the internal HIP C++-to-LLVM IR compiler call.
            Note that you can speficy include paths here.
            Note that the default value ``None`` has a special meaning:
            If ``None`` (the default) is specified, previously specified
            flags are not overriden. Any list that is specified
            overwrites the existing compilation flags.
            Defaults to ``None``.
    """
    from numba.hip import device_init
    from numba.hip.typing_lowering import hipdevicelib

    if extra_cflags:
        hipdevicelib.hipdevicelib.USER_HIP_CFLAGS.clear()
        hipdevicelib.hipdevicelib.USER_HIP_CFLAGS += extra_cflags
    if code and filepath:
        raise KeyError("only one of 'code' and 'filepath' must be specified")
    elif not code and not filepath:
        raise KeyError("one of 'code' and 'filepath' must be specified")
    if filepath:
        with open(filepath, "r") as infile:
            code = infile.read()
    hipdevicelib.hipdevicelib.USER_HIP_EXTENSIONS = code

    # remove the previously registered stubs from the globals
    for k, _ in hipdevicelib.thestubs:
        delattr(device_init, k)
        del globals()[k]
    # reload the hipdevicelib
    hipdevicelib.reload()

    hipdevicelib.__dict__.update(hipdevicelib.thestubs)
    globals().update(hipdevicelib.thestubs)


from . import codegen
from . import compiler

from .compiler import compile_llvm_ir, compile_llvm_ir_for_current_device

compile_ptx = compile_llvm_ir
compile_ptx_for_current_device = compile_llvm_ir_for_current_device

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
