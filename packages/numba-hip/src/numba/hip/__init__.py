# Copyright (c) 2012, Anaconda, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

import sys

from . import api_util  # noqa: F401, E402
from . import hipdrv  # noqa: E402
from . import hipconfig, util  # noqa: F401

# from numba import runtests


# from numba.core import config


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

# -----------------------------------------------
# Derived modules, make local packages submodules
# -----------------------------------------------


cudadrv = hipdrv

sys.modules["numba.hip.hipdrv"] = hipdrv
for _name, _mod in list(sys.modules.items()):
    if _name.startswith("numba.hip.hipdrv"):
        sys.modules[_name.replace("numba.hip.hipdrv", "numba.hip.cudadrv")] = (
            _mod
        )


from . import codegen  # noqa: F401, E402
from . import compiler  # noqa: F401, E402
from . import decorators  # noqa: F401, E402
from . import descriptor  # noqa: F401, E402
from . import dispatcher  # noqa: F401, E402
from . import kernels  # noqa: F401, E402
from . import target  # noqa: F401, E402
from . import testing  # noqa: F401, E402
from . import tests  # noqa: F401, E402
from . import api, args, errors, hipdecl, hipimpl  # noqa: F401, E402
from .compiler import compile_llvm_ir  # noqa: F401, E402
from .compiler import compile_llvm_ir_for_current_device  # noqa: F401, E402
from .compiler import compile_ptx  # noqa: F401, E402
from .compiler import compile_ptx_for_current_device  # noqa: F401, E402

# Other
from .device_init import *  # noqa: F401, F403, E402
from .device_init import _auto_device  # noqa: F401, E402

cudadecl = hipdecl
cudaimpl = hipimpl
sys.modules["numba.hip.cudadecl"] = hipdecl
sys.modules["numba.hip.cudaimpl"] = hipimpl

# HIP C++ extensions


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
        (TODO(HIP/AMD) workaround this limitation by allowing to supply a pointer intent callback/map.
        Alternatively, user could cast pointers to 64-bit integers in Numba HIP Python code and manually cast
        back to pointer types in user function body.)
        (TODO(HIP/AMD) support arrays by providing intrinsic that gets pointer as 64-bit int from (array|struct)( element)?)
        (TODO(HIP/AMD) add some sort of filtering, likely not all functions in a translation unit should be wrapped.
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
        hipdevicelib._hipdevicelib.USER_HIP_CFLAGS.clear()
        hipdevicelib._hipdevicelib.USER_HIP_CFLAGS += extra_cflags
    if code and filepath:
        raise KeyError("only one of 'code' and 'filepath' must be specified")
    elif not code and not filepath:
        raise KeyError("one of 'code' and 'filepath' must be specified")
    if filepath:
        with open(filepath, "r") as infile:
            code = infile.read()
    hipdevicelib._hipdevicelib.USER_HIP_EXTENSIONS = code

    # remove the previously registered stubs from the globals
    for k, _ in hipdevicelib.thestubs:
        delattr(device_init, k)
        del globals()[k]
    # reload the hipdevicelib
    hipdevicelib.reload()

    hipdevicelib.__dict__.update(hipdevicelib.thestubs)
    globals().update(hipdevicelib.thestubs)


def pose_as_cuda():
    """Delegate all 'numba.cuda*' and 'from numba import cuda' imports to 'numba.hip'.

    After calling this function, you can write:

    ```python
    from numba import cuda
    # same now as 'from numba import hip'

    import numba.cuda.cudadrv
    # same now as 'import numba.hip.hipdrv'.

    # ...
    ```

    Deregisters all ``numba.cuda*`` modules from
    the ``sys.modules`` registry and puts
    all ``numba.hip*`` modules with "numba.cuda" prefix into it
    instead.

    Further replaces the "cuda" attribute value of ``sys.modules["numba"]``
    by ``sys.modules["numba.hip"]``.

    Warning:
        Cannot be undone.
    """
    import sys

    numba_cuda_modules = [
        name for name in sys.modules.keys() if name.startswith("numba.cuda")
    ]
    numba_hip_modules = [
        name for name in sys.modules.keys() if name.startswith("numba.hip")
    ]
    for mod in numba_cuda_modules:
        del sys.modules[mod]
    for mod in numba_hip_modules:
        sys.modules[mod.replace("numba.hip", "numba.cuda")] = sys.modules[mod]
    setattr(sys.modules["numba"], "cuda", sys.modules["numba.hip"])

    # compatibility with dependencies (such as RMM memory allocator for Numba)
    from numba import config

    config.CUDA_USE_NVIDIA_BINDING = True


# clean up
del sys
