# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

"""hip-python - Backward compatibility package for ROCm Python bindings.

This package provides backward compatibility with the old hip-python package structure.
It re-exports modules from the new rocm.bindings namespace.

New code should use: from rocm.bindings import hip, hiprtc, hipblas, etc.
Old code continues to work: from hip import hip, hiprtc, hipblas, etc.

`hip`, `hiprtc` and `hip._util` come with this wheel's hard dependencies.
Every other binding is imported below if its wheel is installed, and its name
stays unbound otherwise.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import importlib.metadata
import importlib.util

# Importing `util.types` binds it on its parent, which is what makes the
# pre-7.14 spelling `hip._util.types.Pointer` resolve.
from rocm.bindings import hip, hiprtc, hiprtc_pyext
from rocm.bindings import util as _util
from rocm.bindings.util import types as _util_types
from rocm.version import (
    HIP_VERSION,
    HIP_VERSION_NAME,
    HIP_VERSION_TUPLE,
    ROCM_VERSION,
    ROCM_VERSION_NAME,
    ROCM_VERSION_TUPLE,
    hip_version_name,
    hip_version_tuple,
    rocm_version_name,
    rocm_version_tuple,
)

# hip-python 3.x carried the HIPRTC extensions as `hip.hiprtc.ext`.
hiprtc.ext = hiprtc_pyext

# scikit-build derives the version from the rendered VERSION file, so the
# installed distribution metadata is the only place to read it back from.
for _dist in ("hip-python", "rocm-bindings-hip"):
    try:
        VERSION = __version__ = importlib.metadata.version(_dist)
        break
    except importlib.metadata.PackageNotFoundError:
        continue
else:
    VERSION = __version__ = "0+unknown"  # never installed, a source tree


def _reraise_unless_missing(err, name):
    """Ignore an uninstalled binding, keep every other import failure loud.

    The two are told apart by asking the finder: `from rocm.bindings import x`
    reports an absent submodule as a plain ImportError naming the package, so
    the exception itself does not carry the distinction.
    """
    if importlib.util.find_spec(f"rocm.bindings.{name}") is not None:
        raise err


# `pip install hip-python[libraries]`
try:
    from rocm.bindings import hipblas
except ImportError as err:
    _reraise_unless_missing(err, "hipblas")

try:
    from rocm.bindings import hipblaslt
except ImportError as err:
    _reraise_unless_missing(err, "hipblaslt")

try:
    from rocm.bindings import hipdnn_backend
except ImportError as err:
    _reraise_unless_missing(err, "hipdnn_backend")

try:
    from rocm.bindings import hipfft
except ImportError as err:
    _reraise_unless_missing(err, "hipfft")

try:
    from rocm.bindings import hiprand
except ImportError as err:
    _reraise_unless_missing(err, "hiprand")

try:
    from rocm.bindings import hipsolver
except ImportError as err:
    _reraise_unless_missing(err, "hipsolver")

try:
    from rocm.bindings import hipsparse
except ImportError as err:
    _reraise_unless_missing(err, "hipsparse")

try:
    from rocm.bindings import hipsparselt
except ImportError as err:
    _reraise_unless_missing(err, "hipsparselt")

try:
    from rocm.bindings import hiptensor
except ImportError as err:
    _reraise_unless_missing(err, "hiptensor")

# `pip install hip-python[systems]`
try:
    from rocm.bindings import amdsmi
except ImportError as err:
    _reraise_unless_missing(err, "amdsmi")

try:
    from rocm.bindings import hipfile
except ImportError as err:
    _reraise_unless_missing(err, "hipfile")

try:
    from rocm.bindings import rccl
except ImportError as err:
    _reraise_unless_missing(err, "rccl")

try:
    from rocm.bindings import roctx
except ImportError as err:
    _reraise_unless_missing(err, "roctx")

# `pip install hip-python[compiler]`
try:
    from rocm.bindings import amd_comgr
except ImportError as err:
    _reraise_unless_missing(err, "amd_comgr")
