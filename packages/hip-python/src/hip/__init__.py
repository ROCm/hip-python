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
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import importlib.metadata

# VERSION/__version__ are sourced lazily from the installed distribution
# metadata (scikit-build derives the version from the rendered VERSION file).
_dynamic_version_attrs = {
    "VERSION",
    "__version__",
}

# Re-export the ROCm/HIP version metadata from the canonical, generated
# source of truth (rocm-bindings-core renders rocm/version.py at codegen
# time). rocm-bindings-core is a hard dependency, so a failed import here
# is a real, loud error rather than something to paper over.
from rocm.version import (
    ROCM_VERSION,
    ROCM_VERSION_NAME,
    rocm_version_name,
    ROCM_VERSION_TUPLE,
    rocm_version_tuple,
    HIP_VERSION,
    HIP_VERSION_NAME,
    hip_version_name,
    HIP_VERSION_TUPLE,
    hip_version_tuple,
)


def __getattr__(name):
    """Lazy-load version attributes and re-export modules from rocm.bindings."""
    # Check if it's a version attribute
    if name in _dynamic_version_attrs:
        value = None
        for _dist in ("hip-python", "rocm-bindings-hip"):
            try:
                value = importlib.metadata.version(_dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        if value is None:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        globals()["VERSION"] = value
        globals()["__version__"] = value
        return value

    # Try to import from rocm.bindings namespace
    # First try rocm.bindings.{name} (for hip, hiprtc)
    module = __import__(f"rocm.bindings.{name}", fromlist=[name])
    globals()[name] = module

    # Special case: attach hiprtc_pyext as hiprtc.ext
    if name == "hiprtc":
        try:
            from rocm.bindings.hip import hiprtc_pyext
            setattr(module, "ext", hiprtc_pyext)
        except ImportError:
            pass

    return module
